#ifndef BACKEND_H
#define BACKEND_H

typedef enum {
    BACKEND_NAIVE = 0,
    BACKEND_IKJ = 1,
    BACKEND_BLOCKED = 2
} BackendType;

void BackendSetType(BackendType type);
BackendType BackendGetType(void);
const char *BackendName(BackendType type);

#endif
