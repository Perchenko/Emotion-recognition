#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "decode.h"
#include "logging.h"
#include <malloc.h>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = decodeplugin::INPUT_H;  // H, W must be able to  be divided by 32.
static const int INPUT_W = decodeplugin::INPUT_W;
static const int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2  * 15 + 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;
static Logger gLogger;


cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h* img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::Rect get_rect_adapt_landmark(cv::Mat& img, float bbox[4], float lmk[10]) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] / r_w;
        r = bbox[2] / r_w;
        t = (bbox[1] - (INPUT_H - r_w * img.rows) / 2) / r_w;
        b = (bbox[3] - (INPUT_H - r_w * img.rows) / 2) / r_w;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] /= r_w;
            lmk[i + 1] = (lmk[i + 1] - (INPUT_H - r_w * img.rows) / 2) / r_w;
        }
    } else {
        l = (bbox[0] - (INPUT_W - r_h * img.cols) / 2) / r_h;
        r = (bbox[2] - (INPUT_W - r_h * img.cols) / 2) / r_h;
        t = bbox[1] / r_h;
        b = bbox[3] / r_h;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] = (lmk[i] - (INPUT_W - r_h * img.cols) / 2) / r_h;
            lmk[i + 1] /= r_h;
        }
    }
    return cv::Rect(l, t, r-l, b-t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0], rbox[0]), //left
        std::min(lbox[2], rbox[2]), //right
        std::max(lbox[1], rbox[1]), //top
        std::min(lbox[3], rbox[3]), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) -interBoxS + 0.000001f);
}

bool cmp(const decodeplugin::Detection& a, const decodeplugin::Detection& b) {
    return a.class_confidence > b.class_confidence;
}

void nms(std::vector<decodeplugin::Detection>& res, float *output, float nms_thresh = 0.4) {
    std::vector<decodeplugin::Detection> dets;
    for (int i = 0; i < output[0]; i++) {
        if (output[15 * i + 1 + 4] <= 0.1) continue;
        decodeplugin::Detection det;
        memcpy(&det, &output[15 * i + 1], sizeof(decodeplugin::Detection));
        dets.push_back(det);
    }
    std::sort(dets.begin(), dets.end(), cmp);
    if (dets.size() > 5000) dets.erase(dets.begin() + 5000, dets.end());
    for (size_t m = 0; m < dets.size(); ++m) {
        auto& item = dets[m];
        res.push_back(item);
        //std::cout << item.class_confidence << " bbox " << item.bbox[0] << ", " << item.bbox[1] << ", " << item.bbox[2] << ", " << item.bbox[3] << std::endl;
        for (size_t n = m + 1; n < dets.size(); ++n) {
            if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                dets.erase(dets.begin()+n);
                --n;
            }
        }
    }
}

// Load weights from files
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

Weights getWeights(std::map<std::string, Weights>& weightMap, std::string key) {
    if (weightMap.count(key) != 1) {
        std::cerr << key << " not existed in weight map, fatal error!!!" << std::endl;
        exit(-1);
    }
    return weightMap[key];
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (stride != 1 || inch != outch * 4) {
        IConvolutionLayer* conv4 = network->addConvolutionNd(input, outch * 4, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv4);
        conv4->setStrideNd(DimsHW{stride, stride});

        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

ILayer* conv_bn_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int kernelsize, int stride, int padding, bool userelu, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{kernelsize, kernelsize}, getWeights(weightMap, lname + ".0.weight"), emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{padding, padding});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);

    if (!userelu) return bn1;

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    return relu1;
}

IActivationLayer* ssh(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname) {
    auto conv3x3 = conv_bn_relu(network, weightMap, input, 256 / 2, 3, 1, 1, false, lname + ".conv3X3");
    auto conv5x5_1 = conv_bn_relu(network, weightMap, input, 256 / 4, 3, 1, 1, true, lname + ".conv5X5_1");
    auto conv5x5 = conv_bn_relu(network, weightMap, *conv5x5_1->getOutput(0), 256 / 4, 3, 1, 1, false, lname + ".conv5X5_2");
    auto conv7x7 = conv_bn_relu(network, weightMap, *conv5x5_1->getOutput(0), 256 / 4, 3, 1, 1, true, lname + ".conv7X7_2");
    conv7x7 = conv_bn_relu(network, weightMap, *conv7x7->getOutput(0), 256 / 4, 3, 1, 1, false, lname + ".conv7x7_3");
    ITensor* inputTensors[] = {conv3x3->getOutput(0), conv5x5->getOutput(0), conv7x7->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 3);
    IActivationLayer* relu1 = network->addActivation(*cat->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    return relu1;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("./tensorrtx/retinaface/retinaface_r50.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // ------------- backbone resnet50 ---------------
    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["body.conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "body.bn1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer* x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "body.layer1.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "body.layer1.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "body.layer1.2.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2, "body.layer2.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.3.");
    IActivationLayer* layer2 = x;

    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2, "body.layer3.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.3.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.4.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.5.");
    IActivationLayer* layer3 = x;

    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2, "body.layer4.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "body.layer4.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "body.layer4.2.");
    IActivationLayer* layer4 = x;

    // ------------- FPN ---------------
    auto output1 = conv_bn_relu(network, weightMap, *layer2->getOutput(0), 256, 1, 1, 0, true, "fpn.output1");
    auto output2 = conv_bn_relu(network, weightMap, *layer3->getOutput(0), 256, 1, 1, 0, true, "fpn.output2");
    auto output3 = conv_bn_relu(network, weightMap, *layer4->getOutput(0), 256, 1, 1, 0, true, "fpn.output3");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* up3 = network->addDeconvolutionNd(*output3->getOutput(0), 256, DimsHW{2, 2}, deconvwts, emptywts);
    assert(up3);
    up3->setStrideNd(DimsHW{2, 2});
    up3->setNbGroups(256);
    weightMap["up3"] = deconvwts;

    output2 = network->addElementWise(*output2->getOutput(0), *up3->getOutput(0), ElementWiseOperation::kSUM);
    output2 = conv_bn_relu(network, weightMap, *output2->getOutput(0), 256, 3, 1, 1, true, "fpn.merge2");

    IDeconvolutionLayer* up2 = network->addDeconvolutionNd(*output2->getOutput(0), 256, DimsHW{2, 2}, deconvwts, emptywts);
    assert(up2);
    up2->setStrideNd(DimsHW{2, 2});
    up2->setNbGroups(256);
    output1 = network->addElementWise(*output1->getOutput(0), *up2->getOutput(0), ElementWiseOperation::kSUM);
    output1 = conv_bn_relu(network, weightMap, *output1->getOutput(0), 256, 3, 1, 1, true, "fpn.merge1");

    // ------------- SSH ---------------
    auto ssh1 = ssh(network, weightMap, *output1->getOutput(0), "ssh1");
    auto ssh2 = ssh(network, weightMap, *output2->getOutput(0), "ssh2");
    auto ssh3 = ssh(network, weightMap, *output3->getOutput(0), "ssh3");

    // ------------- Head ---------------
    auto bbox_head1 = network->addConvolutionNd(*ssh1->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.0.conv1x1.weight"], weightMap["BboxHead.0.conv1x1.bias"]);
    auto bbox_head2 = network->addConvolutionNd(*ssh2->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.1.conv1x1.weight"], weightMap["BboxHead.1.conv1x1.bias"]);
    auto bbox_head3 = network->addConvolutionNd(*ssh3->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.2.conv1x1.weight"], weightMap["BboxHead.2.conv1x1.bias"]);

    auto cls_head1 = network->addConvolutionNd(*ssh1->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.0.conv1x1.weight"], weightMap["ClassHead.0.conv1x1.bias"]);
    auto cls_head2 = network->addConvolutionNd(*ssh2->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.1.conv1x1.weight"], weightMap["ClassHead.1.conv1x1.bias"]);
    auto cls_head3 = network->addConvolutionNd(*ssh3->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.2.conv1x1.weight"], weightMap["ClassHead.2.conv1x1.bias"]);

    auto lmk_head1 = network->addConvolutionNd(*ssh1->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.0.conv1x1.weight"], weightMap["LandmarkHead.0.conv1x1.bias"]);
    auto lmk_head2 = network->addConvolutionNd(*ssh2->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.1.conv1x1.weight"], weightMap["LandmarkHead.1.conv1x1.bias"]);
    auto lmk_head3 = network->addConvolutionNd(*ssh3->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.2.conv1x1.weight"], weightMap["LandmarkHead.2.conv1x1.bias"]);

    // ------------- Decode bbox, conf, landmark ---------------
    ITensor* inputTensors1[] = {bbox_head1->getOutput(0), cls_head1->getOutput(0), lmk_head1->getOutput(0)};
    auto cat1 = network->addConcatenation(inputTensors1, 3);
    ITensor* inputTensors2[] = {bbox_head2->getOutput(0), cls_head2->getOutput(0), lmk_head2->getOutput(0)};
    auto cat2 = network->addConcatenation(inputTensors2, 3);
    ITensor* inputTensors3[] = {bbox_head3->getOutput(0), cls_head3->getOutput(0), lmk_head3->getOutput(0)};
    auto cat3 = network->addConcatenation(inputTensors3, 3);

    auto creator = getPluginRegistry()->getPluginCreator("Decode_TRT", "1");
    PluginFieldCollection pfc;
    IPluginV2 *pluginObj = creator->createPlugin("decode", &pfc);
    ITensor* inputTensors[] = {cat1->getOutput(0), cat2->getOutput(0), cat3->getOutput(0)};
    auto decodelayer = network->addPluginV2(inputTensors, 3, *pluginObj);
    assert(decodelayer);

    decodelayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*decodelayer->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
        mem.second.values = NULL;
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext* context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context->getEngine();
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    
}


struct bbox {
    float box[4];
    float landmarks[10];
    float conf;
};

class Detector {
public:
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    std::vector<decodeplugin::Detection> res;
    
    Detector() {
        
    }

    void init() {
        cudaSetDevice(DEVICE);
        // create a model using the API directly and serialize it to a stream
        char *trtModelStream{nullptr};
        size_t size{0};

        // serialize model to plan file
        IHostMemory* modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("retina_r50.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            // return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();

        // deserialize plan file and run inference
        std::ifstream file("retina_r50.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }

        runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        engine = runtime->deserializeCudaEngine(trtModelStream, size);
        //ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
        assert(engine != nullptr);
    //     *context = *engine->createExecutionContext();
        context = engine->createExecutionContext();
        assert(context != nullptr);
    //     void * ptr = &context;
    }
    
    void predict(unsigned char* img_data, float nms_thresh) {
        // prepare input data ---------------------------
    //     IExecutionContext* context = *ptr;
//         IExecutionContext* context = static_cast<IExecutionContext*>(ptr);
        static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
        //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        //    data[i] = 1.0;
        cv::Mat img(cv::Size(INPUT_W, INPUT_H), CV_8UC3, img_data);
//         cv::Mat img = cv::imread("worlds-largest-selfie.jpg");
        cv::Mat pr_img = preprocess_img(img);
        //cv::imwrite("preprocessed.jpg", pr_img);

        // For multi-batch, I feed the same image multiple times.
        // If you want to process different images in a batch, you need adapt it.
        for (int b = 0; b < BATCH_SIZE; b++) {
            float *p_data = &data[b * 3 * INPUT_H * INPUT_W];
            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
                p_data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
                p_data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
            }
        }

        static float prob[BATCH_SIZE * OUTPUT_SIZE];
        doInference(context, data, prob, BATCH_SIZE);

        res.clear();
        for (int b = 0; b < BATCH_SIZE; b++) {
            nms(res, &prob[b * OUTPUT_SIZE], nms_thresh);
//             cv::Mat tmp = img.clone();
//             for (size_t j = 0; j < res.size(); j++) {
//                 if (res[j].class_confidence < 0.1) continue;
//                 cv::Rect r = get_rect_adapt_landmark(tmp, res[j].bbox, res[j].landmark);
//                 cv::rectangle(tmp, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
//                 //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
//                 for (int k = 0; k < 10; k += 2) {
//                     cv::circle(tmp, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
//                 }
//             }
//             cv::imwrite(std::to_string(b) + "_result.jpg", tmp);
        }
    }

};


extern "C" void* init_detector() {
    Detector *d = new Detector;
    d->init();
    return d;
}

extern "C" bbox* detector_predict(void *ptr, int* ptr_size, unsigned char* img_data, float nms_thresh) {
    Detector * d = static_cast<Detector *>(ptr);
    d->predict(img_data, nms_thresh);
    std::vector<decodeplugin::Detection> res = d->res;
    bbox* bboxes = (bbox*)malloc(res.size() * sizeof(bbox));
    for (int i = 0; i < res.size(); i++) {
        for (int k = 0; k < 4; k++) {
            bboxes[i].box[k] = res[i].bbox[k];
        }
        for (int k = 0; k < 10; k++) {
            bboxes[i].landmarks[k] = res[i].landmark[k];
        }
        bboxes[i].conf = res[i].class_confidence;
    }
    *ptr_size = res.size();
    return bboxes;
}
                   

// extern "C" void* init_detector() {
//     void* ptr = new int;
// //     *ptr = 5;
//     std::cout << ptr << std::endl;
//     std::cout << &ptr << std::endl;
//     return ptr;
// }

// extern "C" void detector_predict(void *ptr) {
//     std::cout << ptr << std::endl;
//     std::cout << &ptr << std::endl;
// }



// extern "C" void* prepare() {
//     cudaSetDevice(DEVICE);
//     // create a model using the API directly and serialize it to a stream
//     char *trtModelStream{nullptr};
//     size_t size{0};

//     // serialize model to plan file
//     IHostMemory* modelStream{nullptr};
//     APIToModel(BATCH_SIZE, &modelStream);
//     assert(modelStream != nullptr);

//     std::ofstream p("retina_r50.engine", std::ios::binary);
//     if (!p) {
//         std::cerr << "could not open plan output file" << std::endl;
//         // return -1;
//     }
//     p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
//     modelStream->destroy();

//     // deserialize plan file and run inference
//     std::ifstream file("retina_r50.engine", std::ios::binary);
//     if (file.good()) {
//         file.seekg(0, file.end);
//         size = file.tellg();
//         file.seekg(0, file.beg);
//         trtModelStream = new char[size];
//         assert(trtModelStream);
//         file.read(trtModelStream, size);
//         file.close();
//     }

//     runtime = createInferRuntime(gLogger);
//     assert(runtime != nullptr);
//     engine = runtime->deserializeCudaEngine(trtModelStream, size);
//     //ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
//     assert(engine != nullptr);
// //     *context = *engine->createExecutionContext();
//     context = engine->createExecutionContext();
//     assert(context != nullptr);
// //     void * ptr = &context;
    
//     std::cout << context << std::endl;
//     std::cout << &context << std::endl;
//     return context;
// }

// extern "C" std::vector<decodeplugin::Detection> forward(void* ptr){//IExecutionContext* context){//, unsigned char *image_data) {
//     std::cout << ptr << std::endl;
//     std::cout << &ptr << std::endl;
//     // prepare input data ---------------------------
// //     IExecutionContext* context = *ptr;
//     IExecutionContext* context = static_cast<IExecutionContext*>(ptr);
//     std::cout << context << std::endl;
//     std::cout << &context << std::endl;
//     static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
//     //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
//     //    data[i] = 1.0;
//     std::cout << "Before" << std::endl;
//     cv::Mat img = cv::imread("worlds-largest-selfie.jpg");
// //     cv::Mat img(624, 1024, CV_8UC3, image_data);
//     cv::imwrite(std::to_string(1488) + "_result.jpg", img);
//     std::cout << "After" << std::endl;
//     // cv::Mat img = cv::imread("worlds-largest-selfie.jpg");
//     cv::Mat pr_img = preprocess_img(img);
//     //cv::imwrite("preprocessed.jpg", pr_img);

//     // For multi-batch, I feed the same image multiple times.
//     // If you want to process different images in a batch, you need adapt it.
//     for (int b = 0; b < BATCH_SIZE; b++) {
//         float *p_data = &data[b * 3 * INPUT_H * INPUT_W];
//         for (int i = 0; i < INPUT_H * INPUT_W; i++) {
//             p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
//             p_data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
//             p_data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
//         }
//     }

//     // Run inference
//     static float prob[BATCH_SIZE * OUTPUT_SIZE];
//     // auto start = std::chrono::system_clock::now();
//     std::cout << "111" << std::endl;
//     doInference(context, data, prob, BATCH_SIZE);
//     std::cout << "222" << std::endl;
//     // auto end = std::chrono::system_clock::now();
//     // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


//     for (int b = 0; b < BATCH_SIZE; b++) {
//         std::vector<decodeplugin::Detection> res;
//         nms(res, &prob[b * OUTPUT_SIZE]);

//         // std::cout << "number of detections -> " << prob[b * OUTPUT_SIZE] << std::endl;
//         // std::cout << " -> " << prob[b * OUTPUT_SIZE + 10] << std::endl;
//         // std::cout << "after nms -> " << res.size() << std::endl;
//         cv::Mat tmp = img.clone();
//         for (size_t j = 0; j < res.size(); j++) {
//             if (res[j].class_confidence < 0.1) continue;
//             cv::Rect r = get_rect_adapt_landmark(tmp, res[j].bbox, res[j].landmark);
//             cv::rectangle(tmp, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
//             //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
//             for (int k = 0; k < 10; k += 2) {
//                 cv::circle(tmp, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
//             }
//         }
//         std::cout << "AAA" << std::endl;
//         cv::imwrite(std::to_string(b) + "_result.jpg", tmp);
//         std::cout << "BBB" << std::endl;
//         return res; // WARNING! ONLY FIRST ELEM OF BATCH
//     }
// }

// extern "C" int* func1(int x){
//     std::cout << "START" << std::endl;
//     int *ptr = new int;
//     *ptr = x;
//     std::cout << ptr << std::endl;
//     return ptr;
// }

// extern "C" void func2(int* p){
//     std::cout << p << std::endl;
//     std::cout << *p + 1 << std::endl;
// }

// int y = 0;

// extern "C" int* func1(int x){
//     std::cout << "START" << std::endl;
//     y = x;
//     int *ptr = &y;
//     std::cout << ptr << std::endl;
//     return ptr;
// }

// extern "C" void func2(int* p){
//     std::cout << p << std::endl;
//     std::cout << *p + 1 << std::endl;
// }



int main(int argc, char** argv) {
//     IExecutionContext* context = prepare();
//     cv::Mat img = cv::imread("../worlds-largest-selfie.jpg");
//     std::vector<decodeplugin::Detection> res = forward(context);//, img);
//     std::cout << "SUCCESS" << std::endl;
    
    // if (argc != 2) {
    //     std::cerr << "arguments not right!" << std::endl;
    //     std::cerr << "./retina_r50 -s   // serialize model to plan file" << std::endl;
    //     std::cerr << "./retina_r50 -d   // deserialize plan file and run inference" << std::endl;
    //     return -1;
    // }

    // cudaSetDevice(DEVICE);
    // // create a model using the API directly and serialize it to a stream
    // char *trtModelStream{nullptr};
    // size_t size{0};

    // if (std::string(argv[1]) == "-s") {
    //     IHostMemory* modelStream{nullptr};
    //     APIToModel(BATCH_SIZE, &modelStream);
    //     assert(modelStream != nullptr);

    //     std::ofstream p("retina_r50.engine", std::ios::binary);
    //     if (!p) {
    //         std::cerr << "could not open plan output file" << std::endl;
    //         return -1;
    //     }
    //     p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    //     modelStream->destroy();
    //     return 1;
    // } else if (std::string(argv[1]) == "-d") {
    //     std::ifstream file("retina_r50.engine", std::ios::binary);
    //     if (file.good()) {
    //         file.seekg(0, file.end);
    //         size = file.tellg();
    //         file.seekg(0, file.beg);
    //         trtModelStream = new char[size];
    //         assert(trtModelStream);
    //         file.read(trtModelStream, size);
    //         file.close();
    //     }
    // } else {
    //     return -1;
    // }

    // // prepare input data ---------------------------
    // static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    // //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    // //    data[i] = 1.0;

    // cv::Mat img = cv::imread("worlds-largest-selfie.jpg");
    // cv::Mat pr_img = preprocess_img(img);
    // //cv::imwrite("preprocessed.jpg", pr_img);

    // // For multi-batch, I feed the same image multiple times.
    // // If you want to process different images in a batch, you need adapt it.
    // for (int b = 0; b < BATCH_SIZE; b++) {
    //     float *p_data = &data[b * 3 * INPUT_H * INPUT_W];
    //     for (int i = 0; i < INPUT_H * INPUT_W; i++) {
    //         p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
    //         p_data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
    //         p_data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
    //     }
    // }

    // IRuntime* runtime = createInferRuntime(gLogger);
    // assert(runtime != nullptr);
    // ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    // //ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    // assert(engine != nullptr);
    // IExecutionContext* context = engine->createExecutionContext();
    // assert(context != nullptr);

    // // Run inference
    // static float prob[BATCH_SIZE * OUTPUT_SIZE];
    // auto start = std::chrono::system_clock::now();
    // doInference(*context, data, prob, BATCH_SIZE);
    // auto end = std::chrono::system_clock::now();
    // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // for (int b = 0; b < BATCH_SIZE; b++) {
    //     std::vector<decodeplugin::Detection> res;
    //     nms(res, &prob[b * OUTPUT_SIZE]);
    //     std::cout << "number of detections -> " << prob[b * OUTPUT_SIZE] << std::endl;
    //     std::cout << " -> " << prob[b * OUTPUT_SIZE + 10] << std::endl;
    //     std::cout << "after nms -> " << res.size() << std::endl;
    //     cv::Mat tmp = img.clone();
    //     for (size_t j = 0; j < res.size(); j++) {
    //         if (res[j].class_confidence < 0.1) continue;
    //         cv::Rect r = get_rect_adapt_landmark(tmp, res[j].bbox, res[j].landmark);
    //         cv::rectangle(tmp, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
    //         //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
    //         for (int k = 0; k < 10; k += 2) {
    //             cv::circle(tmp, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
    //         }
    //     }
    //     cv::imwrite(std::to_string(b) + "_result.jpg", tmp);
    // }

    // Destroy the engine
//     context->destroy();
    // engine->destroy();
    // runtime->destroy();

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << i / 10 << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
