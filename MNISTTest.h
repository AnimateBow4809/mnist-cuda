#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
using namespace std;


uint32_t swap_endian(uint32_t val);
void read_mnist_images(const string& filename, float*& images, int& num_images, int& img_size);
void read_mnist_labels(const string& filename, float*& labels, int& num_labels);