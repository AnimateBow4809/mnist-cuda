#include "MNISTTest.h"


uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0x000000FF) |
        ((val >> 8) & 0x0000FF00) |
        ((val << 8) & 0x00FF0000) |
        ((val << 24) & 0xFF000000);
}

void read_mnist_images(const string& filename, float*& images, int& num_images, int& img_size) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    // Read metadata
    uint32_t magic, n_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&n_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    // Convert endianness (MNIST files are big-endian)
    magic = swap_endian(magic);
    n_images = swap_endian(n_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (magic != 2051) {
        cerr << "Invalid MNIST image file: " << filename << endl;
        exit(1);
    }

    num_images = n_images;
    img_size = rows * cols;
    images = new float[num_images * img_size];
    printf("%d nummm", num_images);

    // Read image data
    unsigned char* buffer = new unsigned char[img_size];
    for (int i = 0; i < num_images; i++) {
        file.read(reinterpret_cast<char*>(buffer), img_size);
        for (int j = 0; j < img_size; j++) {
            images[i * img_size + j] = buffer[j] / 255.0f;  // Normalize to [0,1]
        }
    }
    delete[] buffer;
    file.close();
}

void read_mnist_labels(const string& filename, float*& labels, int& num_labels) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    // Read metadata
    uint32_t magic, n_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&n_labels), 4);

    // Convert endianness
    magic = swap_endian(magic);
    n_labels = swap_endian(n_labels);

    if (magic != 2049) {
        cerr << "Invalid MNIST label file: " << filename << endl;
        exit(1);
    }

    num_labels = n_labels;
    labels = new float[num_labels];
    printf("%d nummm", num_labels);

    // Read label data
    unsigned char label;
    for (int i = 0; i < num_labels; i++) {
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = static_cast<float>(label);
    }
    file.close();
}

