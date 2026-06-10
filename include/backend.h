#ifndef BACKEND_H
#define BACKEND_H

typedef enum {
    BACKEND_NAIVE = 0,
    BACKEND_IKJ = 1,
    BACKEND_BLOCKED = 2,
    BACKEND_SIMD = 3,
    BACKEND_THREADS = 4,
    BACKEND_BLAS = 5
} BackendType;

void BackendSetType(BackendType type);
BackendType BackendGetType(void);
const char *BackendName(BackendType type);
int BackendIsAvailable(BackendType type);
BackendType BackendFallback(BackendType type);

#endif
