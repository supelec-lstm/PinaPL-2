//
//  idxParser.cpp
//  PinaPL
//

#include <iostream>
#include <fstream>
#include <ios>
#include <sstream>
#include <zlib.h>
#include <assert.h>

#include "idxParser.hpp"


using namespace std;

void debug(string &title, string &result);

IdxParser::IdxParser() {
    // IdxParser constructor
}

gzFile IdxParser::importGzFile(string& path) {
    // imports a .gz file and returns a gzFile object
    const char * pathChared = path.c_str();
    gzFile file = gzopen(pathChared, "rb");
    if (!file) {
        cout << "Error: could not open file" << endl;
        assert(file);
    }
    return file;
}

vector<vector<double>> IdxParser::importMNISTImages(string path) {
    // reads an IDX3 .gz file (like MNIST image files) and returns a vector of images (pixels for an image are stored in a single continuous vector)
    // (it is in fact treated like a IDX2 file, considering the images have only one dimension)

    // import images file
    gzFile file = importGzFile(path);
    
    // magic number
    uint8_t magicNumberChared[4];
    gzread(file, &magicNumberChared, sizeof(magicNumberChared));
//    int magicNumber =  uint32_t(magicNumberChared[0] << 24 | magicNumberChared[1] << 16 | magicNumberChared[2] << 8 | magicNumberChared[3]);
    
    // first IDX3 dimension: number of images
    uint8_t imageCountChared[4];
    gzread(file, &imageCountChared, sizeof(imageCountChared));
    int imageCount = uint32_t(imageCountChared[0] << 24 | imageCountChared[1] << 16 | imageCountChared[2] << 8 | imageCountChared[3]);
    // second IDX3 dimension: number of rows
    uint8_t rowCountChared[4];
    gzread(file, &rowCountChared, sizeof(rowCountChared));
    int rowCount = uint32_t(rowCountChared[0] << 24 | rowCountChared[1] << 16 | rowCountChared[2] << 8 | rowCountChared[3]);
    // third IDX3 dimension: number of columns
    uint8_t columnCountChared[4];
    gzread(file, &columnCountChared, sizeof(columnCountChared));
    int columnCount = uint32_t(columnCountChared[0] << 24 | columnCountChared[1] << 16 | columnCountChared[2] << 8 | columnCountChared[3]);
    
    vector<vector<double>> output;
    
    for (int imageIndex = 0; imageIndex < imageCount; imageIndex++) {
        vector<double> image;
        for (int pixelIndex = 0; pixelIndex < rowCount*columnCount; pixelIndex++) {
            uint8_t pixelLevelChared;
            gzread(file, &pixelLevelChared, sizeof(pixelLevelChared));
            int pixelLevel = uint8_t(pixelLevelChared);
            image.push_back(static_cast<double>(pixelLevel) / 255.0);
        }
        output.push_back(image);
    }
    
    return(output);
}

vector<long> IdxParser::importMNISTLabels(string path) {
    // reads an IDX1 .gz file (like MNIST label files) and returns a vector of labels

    // import labels file
    gzFile file = importGzFile(path);
    
    // magic number
    uint8_t magicNumberChared[4];
    gzread(file, &magicNumberChared, sizeof(magicNumberChared));
//    int magicNumber =  uint32_t(magicNumberChared[0] << 24 | magicNumberChared[1] << 16 | magicNumberChared[2] << 8 | magicNumberChared[3]);

    // first and only IDX1 dimension: number of labels
    uint8_t labelCountChared[4];
    gzread(file, &labelCountChared, sizeof(labelCountChared));
    int labelCount = uint32_t(labelCountChared[0] << 24 | labelCountChared[1] << 16 | labelCountChared[2] << 8 | labelCountChared[3]);
    
    vector<long> output;
    
    for (int labelIndex = 0; labelIndex < labelCount; labelIndex++) {
        uint8_t labelChared;
        gzread(file, &labelChared, sizeof(labelChared));
        int label = uint8_t(labelChared);
        output.push_back(label);
    }
    
    return(output);
}
