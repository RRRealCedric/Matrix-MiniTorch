#ifndef DIGITS_DATASET_H
#define DIGITS_DATASET_H

#include "tensor.h"

typedef struct {
    int sample_count;
    Tensor inputs;
    Tensor targets;
    int *labels;
} DigitsDataset;

void DigitsDatasetInit(DigitsDataset *dataset);
void DigitsDatasetFree(DigitsDataset *dataset);
MatrixError DigitsDatasetLoadCSV(const char *path, DigitsDataset *dataset);

#endif
