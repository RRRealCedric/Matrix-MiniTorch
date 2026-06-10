#include "digits_dataset.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIGITS_INPUT_DIM 784
#define DIGITS_CLASS_COUNT 10

static int IsHeaderLine(const char *line)
{
    while (*line != '\0' && isspace((unsigned char)*line)) {
        ++line;
    }
    return *line != '\0' && !isdigit((unsigned char)*line) && *line != '-';
}

static MatrixError ParseSampleLine(const char *line, REAL *input_row, REAL *target_row, int *label_out)
{
    char *buffer;
    char *token;
    char *end_ptr;
    long value;
    int label;
    int column = 0;

    if (line == NULL || input_row == NULL || target_row == NULL || label_out == NULL) {
        return MATRIX_ERROR_NULL_POINTER;
    }

    buffer = (char *)malloc(strlen(line) + 1U);
    if (buffer == NULL) {
        return MATRIX_ERROR_ALLOC_FAILED;
    }
    strcpy(buffer, line);

    for (int i = 0; i < DIGITS_CLASS_COUNT; ++i) {
        target_row[i] = 0.0;
    }

    token = strtok(buffer, ",\r\n");
    if (token == NULL) {
        free(buffer);
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }

    value = strtol(token, &end_ptr, 10);
    if (*token == '\0' || *end_ptr != '\0' || value < 0 || value >= DIGITS_CLASS_COUNT) {
        free(buffer);
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }
    label = (int)value;
    target_row[label] = 1.0;
    *label_out = label;

    while ((token = strtok(NULL, ",\r\n")) != NULL) {
        if (column >= DIGITS_INPUT_DIM) {
            free(buffer);
            return MATRIX_ERROR_SIZE_MISMATCH;
        }

        value = strtol(token, &end_ptr, 10);
        if (*token == '\0' || *end_ptr != '\0' || value < 0 || value > 255) {
            free(buffer);
            return MATRIX_ERROR_INVALID_ARGUMENT;
        }

        input_row[column++] = (REAL)value / 255.0;
    }

    free(buffer);
    return column == DIGITS_INPUT_DIM ? MATRIX_SUCCESS : MATRIX_ERROR_SIZE_MISMATCH;
}

void DigitsDatasetInit(DigitsDataset *dataset)
{
    if (dataset == NULL) {
        return;
    }
    dataset->sample_count = 0;
    dataset->labels = NULL;
    TensorInit(&dataset->inputs);
    TensorInit(&dataset->targets);
}

void DigitsDatasetFree(DigitsDataset *dataset)
{
    if (dataset == NULL) {
        return;
    }
    free(dataset->labels);
    dataset->labels = NULL;
    dataset->sample_count = 0;
    TensorFree(&dataset->inputs);
    TensorFree(&dataset->targets);
}

MatrixError DigitsDatasetLoadCSV(const char *path, DigitsDataset *dataset)
{
    FILE *file;
    char line[32768];
    int sample_count = 0;
    int row = 0;
    MatrixError error;

    if (path == NULL || dataset == NULL) {
        return MATRIX_ERROR_NULL_POINTER;
    }

    DigitsDatasetFree(dataset);
    DigitsDatasetInit(dataset);

    file = fopen(path, "r");
    if (file == NULL) {
        return MATRIX_ERROR_INVALID_ARGUMENT;
    }

    while (fgets(line, (int)sizeof(line), file) != NULL) {
        if (line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        if (sample_count == 0 && IsHeaderLine(line)) {
            continue;
        }
        ++sample_count;
    }

    if (sample_count <= 0) {
        fclose(file);
        return MATRIX_ERROR_INVALID_SIZE;
    }

    error = TensorCreate2D(&dataset->inputs, sample_count, DIGITS_INPUT_DIM);
    if (error != MATRIX_SUCCESS) {
        fclose(file);
        return error;
    }

    error = TensorCreate2D(&dataset->targets, sample_count, DIGITS_CLASS_COUNT);
    if (error != MATRIX_SUCCESS) {
        fclose(file);
        DigitsDatasetFree(dataset);
        return error;
    }

    dataset->labels = (int *)malloc((size_t)sample_count * sizeof(int));
    if (dataset->labels == NULL) {
        fclose(file);
        DigitsDatasetFree(dataset);
        return MATRIX_ERROR_ALLOC_FAILED;
    }

    rewind(file);
    while (fgets(line, (int)sizeof(line), file) != NULL) {
        REAL *input_row;
        REAL *target_row;

        if (line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        if (row == 0 && IsHeaderLine(line)) {
            continue;
        }

        input_row = dataset->inputs.data + (size_t)row * (size_t)DIGITS_INPUT_DIM;
        target_row = dataset->targets.data + (size_t)row * (size_t)DIGITS_CLASS_COUNT;
        error = ParseSampleLine(line, input_row, target_row, &dataset->labels[row]);
        if (error != MATRIX_SUCCESS) {
            fclose(file);
            DigitsDatasetFree(dataset);
            return error;
        }
        ++row;
    }

    fclose(file);
    dataset->sample_count = row;
    return row == sample_count ? MATRIX_SUCCESS : MATRIX_ERROR_INVALID_ARGUMENT;
}
