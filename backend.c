#include "backend.h"

static BackendType current_backend = BACKEND_NAIVE;

void BackendSetType(BackendType type)
{
    if (type == BACKEND_NAIVE || type == BACKEND_IKJ || type == BACKEND_BLOCKED) {
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
        default:
            return "unknown";
    }
}
