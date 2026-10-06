/* One-shot transfer of secondary trace files over Multikernel AF_VSOCK. */
#include <errno.h>
#include <fcntl.h>
#include <linux/vm_sockets.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

enum { PORT = 6003, CHUNK = 32 * 1024 };

static int all_io(int fd, void *buf, size_t len, int sending) {
    size_t done = 0;
    unsigned int retries = 0;
    while (done < len) {
        size_t count = len - done;
        if (count > CHUNK) count = CHUNK;
        ssize_t n = sending ? send(fd, (char *)buf + done, count, 0)
                            : recv(fd, (char *)buf + done, count, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && sending && (errno == ENOSPC || errno == EAGAIN)) {
            if (++retries > 1000) return -1;
            usleep(100);
            continue;
        }
        if (n <= 0) return -1;
        retries = 0;
        done += (size_t)n;
        if (sending && done < len) usleep(100);
    }
    return 0;
}

static int mk_socket(void) {
    int fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    int type = 1;
    if (fd < 0 || setsockopt(fd, AF_VSOCK, 9, &type, sizeof(type)) < 0) {
        perror("Multikernel VSOCK");
        return -1;
    }
    return fd;
}

static int serve(const char *path) {
    int input = open(path, O_RDONLY);
    struct stat st;
    if (input < 0 || fstat(input, &st) || st.st_size < 0) {
        perror("input"); return 1;
    }
    int listener = mk_socket();
    struct sockaddr_vm addr = {.svm_family = AF_VSOCK,
                               .svm_cid = VMADDR_CID_ANY, .svm_port = PORT};
    if (listener < 0 || bind(listener, (void *)&addr, sizeof(addr)) ||
        listen(listener, 1)) { perror("listen"); return 1; }
    puts("VSOCK_FILE_READY"); fflush(stdout);
    int peer = accept(listener, NULL, NULL);
    uint64_t bytes = (uint64_t)st.st_size;
    if (peer < 0 || all_io(peer, &bytes, sizeof(bytes), 1)) return 1;
    char data[CHUNK];
    while (bytes) {
        size_t count = bytes < CHUNK ? (size_t)bytes : CHUNK;
        ssize_t n = read(input, data, count);
        if (n <= 0 || all_io(peer, data, (size_t)n, 1)) return 1;
        bytes -= (uint64_t)n;
    }
    close(peer); close(listener); close(input);
    return 0;
}

static int pull(uint32_t cid, const char *path) {
    int peer = mk_socket();
    struct sockaddr_vm addr = {.svm_family = AF_VSOCK,
                               .svm_cid = cid, .svm_port = PORT};
    if (peer < 0 || connect(peer, (void *)&addr, sizeof(addr))) {
        perror("connect"); return 1;
    }
    uint64_t bytes;
    if (all_io(peer, &bytes, sizeof(bytes), 0) || bytes > (1ULL << 30)) return 1;
    int output = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (output < 0) { perror("output"); return 1; }
    uint64_t remaining = bytes;
    char data[CHUNK];
    while (remaining) {
        size_t count = remaining < CHUNK ? (size_t)remaining : CHUNK;
        if (all_io(peer, data, count, 0)) return 1;
        size_t written = 0;
        while (written < count) {
            ssize_t n = write(output, data + written, count - written);
            if (n <= 0) return 1;
            written += (size_t)n;
        }
        remaining -= count;
    }
    close(output); close(peer);
    printf("received %llu bytes\n", (unsigned long long)bytes);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "server") == 0) return serve(argv[2]);
    if (argc == 4 && strcmp(argv[1], "client") == 0)
        return pull((uint32_t)strtoul(argv[2], NULL, 10), argv[3]);
    fprintf(stderr, "usage: %s server FILE | client CID OUTPUT\n", argv[0]);
    return 2;
}
