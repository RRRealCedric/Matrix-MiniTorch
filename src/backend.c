#include "backend.h"

static BackendType current_backend = BACKEND_NAIVE;

void BackendSetType(BackendType type)
{
    if (type >= BACKEND_NAIVE && type <= BACKEND_BLAS) {
        current_backend = type;
    }
}

BackendType BackendGetType(void)
{
    return current_backend;
}

const char *BackendName(BackendType type)
{
    switch (type) {
        case BACKEND_NAIVE:
            return "naive";
        case BACKEND_IKJ:
            return "ikj";
        case BACKEND_BLOCKED:
            return "blocked";
        case BACKEND_SIMD:
            return "simd";
        case BACKEND_THREADS:
            return "threads";
        case BACKEND_BLAS:
            return "blas";
        default:
            return "unknown";
    }
}

int BackendIsAvailable(BackendType type)
{
    switch (type) {
        case BACKEND_NAIVE:
        case BACKEND_IKJ:
        case BACKEND_BLOCKED:
            return 1;
        case BACKEND_SIMD:
#ifdef MATRIX_ENABLE_SIMD
            return 1;
#else
            return 0;
#endif
        case BACKEND_THREADS:
#ifdef MATRIX_ENABLE_THREADS
            return 1;
#else
            return 0;
#endif
        case BACKEND_BLAS:
#ifdef MATRIX_ENABLE_BLAS
            return 1;
#else
            return 0;
#endif
        default:
            return 0;
    }
}

BackendType BackendFallback(BackendType type)
{
    if (BackendIsAvailable(type)) {
        return type;
    }
    return BACKEND_BLOCKED;
}
