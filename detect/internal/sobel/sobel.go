package sobel

/*
#cgo CFLAGS: -I./cuda-sobel
#cgo LDFLAGS: -L./cuda-sobel -lsobel -Wl,-rpath=./cuda-sobel

#include "sobel.h"
*/
import "C"
import (
	"unsafe"
)

func Apply(data []uint8, w, h int) error {
	status := C.ApplySobel((*C.uint)(unsafe.Pointer(&data[0])), C.int(w), C.int(h))
	_ = status
	return nil
}
