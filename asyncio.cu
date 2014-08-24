#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <gelf.h>
#include <map>
#include <setjmp.h>
#include <string>
#include <string.h>
#include <sstream>
#include <vector>
#include <unistd.h>

#define ASYNCIO_BUFFER_LENGTH (1024 * 1024 * 16)
#define ASYNCIO_STDOUT -1
#define ASYNCIO_UNFORMATTED -1

using namespace std;

struct transaction_t
{
	int unit;
	int format;
	void* func;
	int nitems;
	int offset;
	char* buffer;
};

enum type_t
{
	TYPE_INT,
	TYPE_INT_1D,
	TYPE_INT_2D,
	TYPE_LONG_LONG,
	TYPE_FLOAT,
	TYPE_FLOAT_1D,
	TYPE_DOUBLE,
	TYPE_DOUBLE_1D,
	TYPE_DOUBLE_2D,
	TYPE_DOUBLE_3D,
	TYPE_DOUBLE_4D,
	TYPE_BOOLEAN,
	TYPE_BOOLEAN_1D,
	TYPE_CHAR,
	TYPE_CHAR_1D,
	TYPE_CHAR_2D
};

#ifdef __CUDACC__
#define DEVICE __device__
#define NAMESPACE gpu
#else
#define DEVICE
#define NAMESPACE cpu
#define trap() exit(1)
#endif

namespace NAMESPACE
{
	DEVICE bool asyncio_error = false;
	DEVICE char asyncio_buffer[ASYNCIO_BUFFER_LENGTH];
	DEVICE size_t asyncio_buffer_length = 0;

	DEVICE char* asyncio_pbuffer = NULL;
	DEVICE transaction_t* t_curr = NULL;
	DEVICE int t_curr_nitems = 0;
}

using namespace NAMESPACE;

// On GPU all I/O routines work with thread 0 only.

extern "C" DEVICE void asyncio_begin_default_c(char unit, char format)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (t_curr)
	{
		printf("ERROR: Previous transaction has not been closed correctly\n");
		asyncio_error = true;
		trap();
	}

	if (unit != '*')
	{
		printf("ERROR: Invalid unit specifier: %c\n", unit);
		asyncio_error = true;
		trap();
	}
	if (format != '*')
	{
		printf("ERROR: Invalid format specifier: %c\n", format);
		asyncio_error = true;
		trap();		
	}

	if (!asyncio_pbuffer) asyncio_pbuffer = asyncio_buffer;

	transaction_t t;
	t.unit = ASYNCIO_STDOUT;
	t.format = ASYNCIO_UNFORMATTED;
	t_curr_nitems = 0;
	
	memcpy(asyncio_pbuffer, &t, sizeof(transaction_t));
	t_curr = (transaction_t*)asyncio_pbuffer;
	asyncio_pbuffer += sizeof(transaction_t);
}

extern "C" DEVICE void asyncio_begin_default_format_c(int unit, char format)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (t_curr)
	{
		printf("ERROR: Previous transaction has not been closed correctly\n");
		asyncio_error = true;
		trap();
	}

	if (format != '*')
	{
		printf("ERROR: Invalid format specifier: %c\n", format);
		asyncio_error = true;
		trap();		
	}

	if (!asyncio_pbuffer) asyncio_pbuffer = asyncio_buffer;

	transaction_t t;
	t.unit = unit;
	t.format = ASYNCIO_UNFORMATTED;
	t_curr_nitems = 0;
	
	memcpy(asyncio_pbuffer, &t, sizeof(transaction_t));
	t_curr = (transaction_t*)asyncio_pbuffer;
	asyncio_pbuffer += sizeof(transaction_t);
}

extern "C" DEVICE void asyncio_begin_default_unit_c(char unit, void* func, int format)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (t_curr)
	{
		printf("ERROR: Previous transaction has not been closed correctly\n");
		asyncio_error = true;
		trap();
	}

	if (unit != '*')
	{
		printf("ERROR: Invalid unit specifier: %c\n", unit);
		asyncio_error = true;
		trap();
	}

	if (!asyncio_pbuffer) asyncio_pbuffer = asyncio_buffer;

	transaction_t t;
	t.unit = ASYNCIO_STDOUT;
	t.format = format;
	t.func = func;
	t_curr_nitems = 0;
	
	memcpy(asyncio_pbuffer, &t, sizeof(transaction_t));
	t_curr = (transaction_t*)asyncio_pbuffer;
	asyncio_pbuffer += sizeof(transaction_t);
}

extern "C" DEVICE void asyncio_begin_c(int unit, void* func, int format)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (t_curr)
	{
		printf("ERROR: Previous transaction has not been closed correctly\n");
		asyncio_error = true;
		trap();
	}

	if (!asyncio_pbuffer) asyncio_pbuffer = asyncio_buffer;

	transaction_t t;
	t.unit = unit;
	t.format = format;
	t.func = func;
	t_curr_nitems = 0;
	
	memcpy(asyncio_pbuffer, &t, sizeof(transaction_t));
	t_curr = (transaction_t*)asyncio_pbuffer;
	asyncio_pbuffer += sizeof(transaction_t);
}

extern "C" DEVICE void asyncio_write_integer_c(int val)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_INT;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	*(int*)asyncio_pbuffer = val;
	asyncio_pbuffer += sizeof(int);
	t_curr_nitems++;
}

extern "C" DEVICE void asyncio_write_long_long_c(long long val)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_LONG_LONG;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	memcpy(asyncio_pbuffer, &val, sizeof(long long));
	asyncio_pbuffer += sizeof(long long);
	t_curr_nitems++;
}

extern "C" DEVICE void asyncio_write_float_c(float val)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_FLOAT;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	memcpy(asyncio_pbuffer, &val, sizeof(float));
	asyncio_pbuffer += sizeof(float);
	t_curr_nitems++;
}

extern "C" DEVICE void asyncio_write_double_c(double val)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_DOUBLE;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	memcpy(asyncio_pbuffer, &val, sizeof(double));
	asyncio_pbuffer += sizeof(double);
	t_curr_nitems++;
}

extern "C" DEVICE void asyncio_write_logical_c(bool val)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_BOOLEAN;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	memcpy(asyncio_pbuffer, &val, sizeof(bool));
	asyncio_pbuffer += sizeof(bool);
	t_curr_nitems++;
}

extern "C" DEVICE void asyncio_write_char_c(char* val, int length)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_CHAR;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	memcpy(asyncio_pbuffer, &length, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, val, sizeof(char) * length);
	asyncio_pbuffer += sizeof(char) * length;
	t_curr_nitems++;
}

extern "C" DEVICE void asyncio_write_char_array1d_c(char** val, int dim_1, int* lengths)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	printf("ERROR: not implemented\n");
	asyncio_error = true;
	trap();
}

extern "C" DEVICE void asyncio_write_char_array2d_c(char** val, int dim_1, int dim_2, int* lengths)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	printf("ERROR: not implemented\n");
	asyncio_error = true;
	trap();
}

extern "C" DEVICE void asyncio_write_logical_array1d_c(bool* val, int dim_1)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_BOOLEAN_1D;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	memcpy(asyncio_pbuffer, &dim_1, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, val, sizeof(bool) * dim_1);
	asyncio_pbuffer += sizeof(bool) * dim_1;
	t_curr_nitems++;
}

extern "C" DEVICE void asyncio_write_integer_array1d_c(int* val, int dim_1)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_INT_1D;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	memcpy(asyncio_pbuffer, &dim_1, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, val, sizeof(int) * dim_1);
	asyncio_pbuffer += sizeof(int) * dim_1;
	t_curr_nitems++;
}

extern "C" DEVICE void asyncio_write_integer_array2d_c(int* val, int dim_1, int dim_2)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_INT_2D;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	memcpy(asyncio_pbuffer, &dim_1, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, &dim_2, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, val, sizeof(int) * dim_1 * dim_2);
	asyncio_pbuffer += sizeof(int) * dim_1 * dim_2;
	t_curr_nitems++;
}

extern "C" DEVICE void asyncio_write_float_array1d_c(float* val, int dim_1)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_FLOAT_1D;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	memcpy(asyncio_pbuffer, &dim_1, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, val, sizeof(float) * dim_1);
	asyncio_pbuffer += sizeof(float) * dim_1;
	t_curr_nitems++;
}

extern "C" DEVICE void asyncio_write_double_array1d_c(double* val, int dim_1)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_DOUBLE_1D;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	memcpy(asyncio_pbuffer, &dim_1, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, val, sizeof(double) * dim_1);
	asyncio_pbuffer += sizeof(double) * dim_1;
	t_curr_nitems++;
}

extern "C" DEVICE void asyncio_write_double_array2d_c(double* val, int dim_1, int dim_2)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_DOUBLE_2D;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	memcpy(asyncio_pbuffer, &dim_1, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, &dim_2, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, val, sizeof(double) * dim_1 * dim_2);
	asyncio_pbuffer += sizeof(double) * dim_1 * dim_2;
	t_curr_nitems++;
}

extern "C" DEVICE void asyncio_write_double_array3d_c(double* val, int dim_1, int dim_2, int dim_3)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_DOUBLE_3D;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	memcpy(asyncio_pbuffer, &dim_1, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, &dim_2, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, &dim_3, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, val, sizeof(double) * dim_1 * dim_2 * dim_3);
	asyncio_pbuffer += sizeof(double) * dim_1 * dim_2 * dim_3;
	t_curr_nitems++;
}

extern "C" DEVICE void asyncio_write_double_array4d_c(double* val, int dim_1, int dim_2, int dim_3, int dim_4)
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to write without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	type_t type = TYPE_DOUBLE_3D;
	memcpy(asyncio_pbuffer, &type, sizeof(type_t));
	asyncio_pbuffer += sizeof(type_t);
	memcpy(asyncio_pbuffer, &dim_1, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, &dim_2, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, &dim_3, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, &dim_4, sizeof(int));
	asyncio_pbuffer += sizeof(int);
	memcpy(asyncio_pbuffer, val, sizeof(double) * dim_1 * dim_2 * dim_3 * dim_4);
	asyncio_pbuffer += sizeof(double) * dim_1 * dim_2 * dim_3 * dim_4;
	t_curr_nitems++;
}

extern "C" void asyncio_flush();

extern "C" DEVICE void asyncio_end()
{
#ifdef __CUDACC__
	if (threadIdx.x) return;
#endif
	if (!t_curr)
	{
		printf("ERROR: Attempted to end without an active transaction\n");
		asyncio_error = true;
		trap();
	}

	memcpy(&t_curr->nitems, &t_curr_nitems, sizeof(int));
	t_curr = NULL;
	
	// Save the current buffer length.
	asyncio_buffer_length = (size_t)asyncio_pbuffer - (size_t)asyncio_buffer;

#ifndef __CUDACC__
	// On host we can flush each individual write statement.
	asyncio_flush();
#endif
}

#define CUDA_ERR_CHECK(x)                                   \
    do { cudaError_t err = x;                               \
        if (err != cudaSuccess) {                           \
        printf("CUDA error %d \"%s\" at %s:%d\n",           \
        (int)err, cudaGetErrorString(err),                  \
        __FILE__, __LINE__); exit(1);                       \
    }} while (0);

struct st_parameter_dt;

extern "C" void asyncio_hook_write_default_unformatted();
extern "C" void asyncio_hook_write_default_formatted(size_t, char*);
extern "C" void asyncio_hook_write_unit_unformatted(int);
extern "C" void asyncio_hook_write_unit_formatted(int, size_t, char*);

extern "C" void asyncio_hook_write_integer_array_1d(void*, int);
extern "C" void asyncio_hook_write_integer_array_2d(void*, int, int);
extern "C" void asyncio_hook_write_float_array_1d(void*, int);
extern "C" void asyncio_hook_write_double_array_1d(void*, int);
extern "C" void asyncio_hook_write_double_array_2d(void*, int, int);
extern "C" void asyncio_hook_write_double_array_3d(void*, int, int, int);
extern "C" void asyncio_hook_write_double_array_4d(void*, int, int, int, int);
extern "C" void asyncio_hook_write_boolean_array_1d(void*, int);

static bool inside_hook_write = false;
static bool inside_hook_write_array = false;

static jmp_buf get_st_parameter_jmp;

typedef void (*st_write_callback_t)(transaction_t*, st_parameter_dt*);
static st_write_callback_t callback;
static transaction_t* transaction;
static st_parameter_dt* st_parameter = NULL;

#ifdef DYNAMIC
#define LIBGFORTRAN "libgfortran.so.3"

static void* libgfortran = NULL;

#define bind_lib(lib) \
if (!libgfortran) \
{ \
	libgfortran = dlopen(lib, RTLD_NOW | RTLD_GLOBAL); \
	if (!libgfortran) \
	{ \
		fprintf(stderr, "Error loading %s: %s\n", lib, dlerror()); \
		abort(); \
	} \
}

#define bind_sym(handle, sym, retty, ...) \
typedef retty (*sym##_func_t)(__VA_ARGS__); \
static sym##_func_t sym##_real = NULL; \
if (!sym##_real) \
{ \
	sym##_real = (sym##_func_t)dlsym(handle, #sym); \
	if (!sym##_real) \
	{ \
		fprintf(stderr, "Error loading %s: %s\n", #sym, dlerror()); \
		abort(); \
	} \
}

extern "C" void _gfortran_st_write(st_parameter_dt * stp)
#else
extern "C" void __real__gfortran_st_write(st_parameter_dt * stp);
extern "C" void __wrap__gfortran_st_write(st_parameter_dt * stp)
#endif
{
	// In case of array writing hook, we discard _gfortran_st_write
	// completely.
	if (inside_hook_write_array)
		return;

#ifdef DYNAMIC
	bind_lib(LIBGFORTRAN);
	bind_sym(libgfortran, _gfortran_st_write, void, st_parameter_dt*);
	_gfortran_st_write_real(stp);
#else
	__real__gfortran_st_write(stp);
#endif
	if (inside_hook_write)
	{
		st_parameter = stp;
		callback(transaction, stp);
		longjmp(get_st_parameter_jmp, 1);
	}
}

extern "C" void _gfortran_st_write_done(st_parameter_dt * stp);
#ifdef DYNAMIC
extern "C" void _gfortran_st_write_done(st_parameter_dt * stp)
#else
extern "C" void __real__gfortran_st_write_done(st_parameter_dt * stp);
extern "C" void __wrap__gfortran_st_write_done(st_parameter_dt * stp)
#endif
{
	// In case of array writing hook, we discard _gfortran_st_write_done
	// completely.
	if (inside_hook_write_array)
		return;

#ifdef DYNAMIC
	bind_lib(LIBGFORTRAN);
	bind_sym(libgfortran, _gfortran_st_write_done, void, st_parameter_dt*);
	_gfortran_st_write_done_real(stp);
#else
	__real__gfortran_st_write_done(stp);
#endif
}

#ifdef DYNAMIC
extern "C" void _gfortran_transfer_array_write(st_parameter_dt* stp, void* desc, int kind, size_t charlen)
#else
extern "C" void __real__gfortran_transfer_array_write(st_parameter_dt* stp, void* desc, int kind, size_t charlen);
extern "C" void __wrap__gfortran_transfer_array_write(st_parameter_dt* stp, void* desc, int kind, size_t charlen)
#endif
{
	// In case of array writing hook, we use our own st_parameter
	if (inside_hook_write_array)
		stp = st_parameter;

#ifdef DYNAMIC
	bind_lib(LIBGFORTRAN);
	bind_sym(libgfortran, _gfortran_transfer_array_write, void, st_parameter_dt*, void*, int, size_t);
	_gfortran_transfer_array_write_real(stp, desc, kind, charlen);
#else
	__real__gfortran_transfer_array_write(stp, desc, kind, charlen);
#endif
}

extern "C" void _gfortran_transfer_integer_write(st_parameter_dt *, void *, int);
extern "C" void _gfortran_transfer_real_write(st_parameter_dt *, void *, int);
extern "C" void _gfortran_transfer_logical_write(st_parameter_dt *, void *, int);
extern "C" void _gfortran_transfer_character_write(st_parameter_dt *, void *, int);

static map<void*, string>* pfuncs = NULL, funcs;
static map<string, void*> formats;
static bool funcs_resolved = false;

static void st_write_callback(transaction_t* t, st_parameter_dt* st_parameter)
{
	for (int i = 0, e = t->nitems; i != e; i++)
	{
		type_t type = *(type_t*)(t->buffer + t->offset);
		t->offset += sizeof(type_t);
		void* value = (void*)(t->buffer + t->offset);

		switch (type)
		{
		case TYPE_INT :
			_gfortran_transfer_integer_write(st_parameter, value, sizeof(int));
			t->offset += sizeof(int);
			break;
		case TYPE_INT_1D :
			{
				int* length = (int*)value;
				t->offset += sizeof(int);
				value = (void*)(t->buffer + t->offset);
				inside_hook_write_array = true;
				asyncio_hook_write_integer_array_1d(value, length[0]);
				inside_hook_write_array = false;
				t->offset += sizeof(int) * length[0];
			}
			break;
		case TYPE_INT_2D :
			{
				int* length = (int*)value;
				t->offset += sizeof(int) * 2;
				value = (void*)(t->buffer + t->offset);
				inside_hook_write_array = true;
				asyncio_hook_write_integer_array_2d(value, length[0], length[1]);
				inside_hook_write_array = false;
				t->offset += sizeof(int) * length[0] * length[1];
			}
			break;
		case TYPE_LONG_LONG :
			_gfortran_transfer_integer_write(st_parameter, value, sizeof(long long));
			t->offset += sizeof(long long);
			break;
		case TYPE_FLOAT :
			_gfortran_transfer_real_write(st_parameter, value, sizeof(float));
			t->offset += sizeof(float);
			break;
		case TYPE_FLOAT_1D :
			{
				int* length = (int*)value;
				t->offset += sizeof(int);
				value = (void*)(t->buffer + t->offset);
				inside_hook_write_array = true;
				asyncio_hook_write_float_array_1d(value, length[0]);
				inside_hook_write_array = false;
				t->offset += sizeof(float) * length[0];
			}
			break;
		case TYPE_DOUBLE :
			_gfortran_transfer_real_write(st_parameter, value, sizeof(double));
			t->offset += sizeof(double);
			break;
		case TYPE_DOUBLE_1D :
			{
				int* length = (int*)value;
				t->offset += sizeof(int);
				value = (void*)(t->buffer + t->offset);
				inside_hook_write_array = true;
				asyncio_hook_write_double_array_1d(value, length[0]);
				inside_hook_write_array = false;
				t->offset += sizeof(double) * length[0];
			}
			break;
		case TYPE_DOUBLE_2D :
			{
				int* length = (int*)value;
				t->offset += sizeof(int) * 2;
				value = (void*)(t->buffer + t->offset);
				inside_hook_write_array = true;
				asyncio_hook_write_double_array_2d(value, length[0], length[1]);
				inside_hook_write_array = false;
				t->offset += sizeof(double) * length[0] * length[1];
			}
			break;
		case TYPE_DOUBLE_3D :
			{
				int* length = (int*)value;
				t->offset += sizeof(int) * 3;
				value = (void*)(t->buffer + t->offset);
				inside_hook_write_array = true;
				asyncio_hook_write_double_array_3d(value, length[0], length[1], length[2]);
				inside_hook_write_array = false;
				t->offset += sizeof(double) * length[0] * length[1] * length[2];
			}
			break;
		case TYPE_DOUBLE_4D :
			{
				int* length = (int*)value;
				t->offset += sizeof(int) * 4;
				value = (void*)(t->buffer + t->offset);
				inside_hook_write_array = true;
				asyncio_hook_write_double_array_4d(value, length[0], length[1], length[2], length[3]);
				inside_hook_write_array = false;
				t->offset += sizeof(double) * length[0] * length[1] * length[2] * length[3];
			}
			break;
		case TYPE_BOOLEAN :
			_gfortran_transfer_logical_write(st_parameter, value, sizeof(bool));
			t->offset += sizeof(bool);
			break;
		case TYPE_BOOLEAN_1D :
			{
				int* length = (int*)value;
				t->offset += sizeof(int);
				value = (void*)(t->buffer + t->offset);
				inside_hook_write_array = true;
				asyncio_hook_write_boolean_array_1d(value, length[0]);
				inside_hook_write_array = false;
				t->offset += sizeof(bool) * length[0];
			}
			break;
		case TYPE_CHAR :
			{
				int length = *(int*)value;
				t->offset += sizeof(int);
				value = (void*)(t->buffer + t->offset);
				_gfortran_transfer_character_write(st_parameter, value, sizeof(char) * length);
				t->offset += sizeof(char) * length;
			}
			break;
		case TYPE_CHAR_1D :
		case TYPE_CHAR_2D :
			fprintf(stderr, "ERROR: not implemented\n");
			exit(1);
			break;
		default :
			fprintf(stderr, "ERROR: Unknown data type %d\n", type);
			exit(1);
		}
	}

	_gfortran_st_write_done(st_parameter);
}

#ifdef __CUDACC__

static char* get_format(void* func, int format)
{
	if (!funcs_resolved)
	{
		// 1) Resolve device functions addresses.
		for (map<void*, string>::iterator i = pfuncs->begin(), e = pfuncs->end(); i != e; i++)
		{
			void* gpuAddress = NULL;
			CUDA_ERR_CHECK(cudaGetSymbolAddress(&gpuAddress, (const void*)i->first));
			funcs[gpuAddress] = i->second;
		}
		delete pfuncs;
		pfuncs = &funcs;

		// 2) Find addresses of all __GPUFMT_* variables in host executable.
		int fd = -1;
		Elf *e = NULL;
		try
		{
			if (elf_version(EV_CURRENT) == EV_NONE)
			{
				fprintf(stderr, "Cannot initialize ELF library: %s\n",
					elf_errmsg(-1));
				throw;
			}
			if ((fd = open("/proc/self/exe", O_RDONLY)) < 0)
			{
				fprintf(stderr, "Cannot open self executable\n");
				throw;
			}
			if ((e = elf_begin(fd, ELF_C_READ, e)) == 0) {
				fprintf(stderr, "Cannot read ELF image: %s\n", elf_errmsg(-1));
				throw;
			}
			size_t shstrndx;
			if (elf_getshdrstrndx(e, &shstrndx)) {
				fprintf(stderr, "elf_getshdrstrndx() failed: %s\n", elf_errmsg(-1));
				throw;
			}

			// Locate the symbol table.
			Elf_Scn* scn = elf_nextscn(e, NULL);
			for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++)
			{
				// Get section header.
				GElf_Shdr shdr;
				if (!gelf_getshdr(scn, &shdr))
				{
					fprintf(stderr, "gelf_getshdr() failed: %s\n", elf_errmsg(-1));
					throw;
				}

				// If section is not a symbol table:
				if (shdr.sh_type != SHT_SYMTAB) continue;

				// Load symbols.
				Elf_Data* data = elf_getdata(scn, NULL);
				if (!data)
				{
					fprintf(stderr, "Expected data section for SYMTAB\n");
					throw;
				}
				if (shdr.sh_size && !shdr.sh_entsize)
				{
					fprintf(stderr, "Cannot get the number of symbols\n");
					throw;
				}
				int nsymbols = 0;
				if (shdr.sh_size)
					nsymbols = shdr.sh_size / shdr.sh_entsize;
				int strndx = shdr.sh_link;
				for (int i = 0; i < nsymbols; i++)
				{
					GElf_Sym sym;
					if (!gelf_getsym(data, i, &sym))
					{
						fprintf(stderr, "gelf_getsym() failed: %s\n", elf_errmsg(-1));
						throw;
					}
					char* name = elf_strptr(e, strndx, sym.st_name);
					if (!name)
					{
						fprintf(stderr, "Cannot get the name of %d-th symbol: %s\n", i, elf_errmsg(-1));
						throw;
					}

					if (!strncmp(name, "__GPUFMT_", strlen("__GPUFMT_")))
					{
						// This symbol is a format string - record it.
						name += strlen("__GPUFMT_");
						formats[name] = (void*)(size_t)sym.st_value;
					}
				}
				elf_end(e);
				close(fd);
				e = NULL;

				funcs_resolved = true;

				break;
			}
		
			if (!funcs_resolved)
			{
				fprintf(stderr, "Cannot locate the symbol table of executable\n");
				throw;
			}
		}
		catch (...)
		{
			if (e)
				elf_end(e);
			if (fd >= 0)
				close(fd);
			exit(1);
		}
	}

	map<void*, string>::iterator i = funcs.find((void*)func);
	if (i == funcs.end())
	{
		fprintf(stderr, "ERROR: Unknown function @ %p\n", (void*)func);
		exit(1);
	}
	stringstream svarname;
	svarname << i->second << "_" << format;
	string varname = svarname.str();
	map<string, void*>::iterator j = formats.find(varname);
	if (j == formats.end())
	{
		fprintf(stderr, "ERROR: Undefined format spec \"%s\"\n", varname.c_str());
		exit(1);
	}
	char* result = (char*)j->second;
	return result;
}

extern "C" void CUDARTAPI __real___cudaRegisterVar(
        void **fatCubinHandle,
        char  *hostVar,
        char  *deviceAddress,
  const char  *deviceName,
        int    ext,
        int    size,
        int    constant,
        int    global
);

extern "C" void CUDARTAPI __wrap___cudaRegisterVar(
        void **fatCubinHandle,
        char  *hostVar,
        char  *deviceAddress,
  const char  *deviceName,
        int    ext,
        int    size,
        int    constant,
        int    global
)
{
	// Workaround in case if static funcs map could happen to be
	// initialized later.
	if (!pfuncs)
		pfuncs = new map<void*, string>();

	__real___cudaRegisterVar(
		fatCubinHandle,
		hostVar,
		deviceAddress,
		deviceName,
		ext,
		size,
		constant,
		global);

	if (strncmp(deviceName, "__FUNC_", strlen("__FUNC_")))
		return;

	// This symbol is a function name anchor - record it.
	string& name = pfuncs->operator[]((void*)hostVar);
	name = deviceAddress + strlen("__FUNC_");
}

#else

static char* get_format(void* func, int format)
{
	if (!funcs_resolved)
	{
		// 1) Find addresses of all __GPUFMT_* variables in host executable.
		int fd = -1;
		Elf *e = NULL;
		try
		{
			if (elf_version(EV_CURRENT) == EV_NONE)
			{
				fprintf(stderr, "Cannot initialize ELF library: %s\n",
					elf_errmsg(-1));
				throw;
			}
			if ((fd = open("/proc/self/exe", O_RDONLY)) < 0)
			{
				fprintf(stderr, "Cannot open self executable\n");
				throw;
			}
			if ((e = elf_begin(fd, ELF_C_READ, e)) == 0) {
				fprintf(stderr, "Cannot read ELF image: %s\n", elf_errmsg(-1));
				throw;
			}
			size_t shstrndx;
			if (elf_getshdrstrndx(e, &shstrndx)) {
				fprintf(stderr, "elf_getshdrstrndx() failed: %s\n", elf_errmsg(-1));
				throw;
			}

			// Locate the symbol table.
			Elf_Scn* scn = elf_nextscn(e, NULL);
			for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++)
			{
				// Get section header.
				GElf_Shdr shdr;
				if (!gelf_getshdr(scn, &shdr))
				{
					fprintf(stderr, "gelf_getshdr() failed: %s\n", elf_errmsg(-1));
					throw;
				}

				// If section is not a symbol table:
				if (shdr.sh_type != SHT_SYMTAB) continue;

				// Load symbols.
				Elf_Data* data = elf_getdata(scn, NULL);
				if (!data)
				{
					fprintf(stderr, "Expected data section for SYMTAB\n");
					throw;
				}
				if (shdr.sh_size && !shdr.sh_entsize)
				{
					fprintf(stderr, "Cannot get the number of symbols\n");
					throw;
				}
				int nsymbols = 0;
				if (shdr.sh_size)
					nsymbols = shdr.sh_size / shdr.sh_entsize;
				int strndx = shdr.sh_link;
				for (int i = 0; i < nsymbols; i++)
				{
					GElf_Sym sym;
					if (!gelf_getsym(data, i, &sym))
					{
						fprintf(stderr, "gelf_getsym() failed: %s\n", elf_errmsg(-1));
						throw;
					}
					char* name = elf_strptr(e, strndx, sym.st_name);
					if (!name)
					{
						fprintf(stderr, "Cannot get the name of %d-th symbol: %s\n", i, elf_errmsg(-1));
						throw;
					}

					if (!strncmp(name, "__FUNC_", strlen("__FUNC_")))
					{
						// This symbol is a function name anchor - record it.
						name += strlen("__FUNC_");
						funcs[(void*)(size_t)sym.st_value] = name;
					}

					if (!strncmp(name, "__GPUFMT_", strlen("__GPUFMT_")))
					{
						// This symbol is a format string - record it.
						name += strlen("__GPUFMT_");
						formats[name] = (void*)(size_t)sym.st_value;
					}
				}
				elf_end(e);
				close(fd);
				e = NULL;

				funcs_resolved = true;

				break;
			}
		
			if (!funcs_resolved)
			{
				fprintf(stderr, "Cannot locate the symbol table of executable\n");
				throw;
			}
		}
		catch (...)
		{
			if (e)
				elf_end(e);
			if (fd >= 0)
				close(fd);
			exit(1);
		}
	}

	map<void*, string>::iterator i = funcs.find((void*)func);
	if (i == funcs.end())
	{
		fprintf(stderr, "ERROR: Unknown function @ %p\n", (void*)func);
		exit(1);
	}
	stringstream svarname;
	svarname << i->second << "_" << format;
	string varname = svarname.str();
	map<string, void*>::iterator j = formats.find(varname);
	if (j == formats.end())
	{
		fprintf(stderr, "ERROR: Undefined format spec \"%s\"\n", varname.c_str());
		exit(1);
	}
	char* result = (char*)j->second;
	return result;
}

#endif // __CUDACC__

extern "C" void asyncio_flush()
{
#ifdef __CUDACC__
	// Transfer asyncio error status.
	static bool* pdevice_error = NULL;
	if (!pdevice_error)
		CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&pdevice_error, asyncio_error));
	bool host_error = true;
	CUDA_ERR_CHECK(cudaMemcpy(&host_error, pdevice_error, sizeof(bool),
		cudaMemcpyDeviceToHost));

	// Do nothing, if error status is true.
	if (host_error) return;

	// Transfer asyncio buffer length.
	static size_t* pdevice_length = NULL;
	if (!pdevice_length)
		CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&pdevice_length, asyncio_buffer_length));
	size_t host_length = 0;
	CUDA_ERR_CHECK(cudaMemcpy(&host_length, pdevice_length, sizeof(size_t),
		cudaMemcpyDeviceToHost));
	
	// Do nothing, if buffer length is zero.
	if (host_length == 0)
	{
		CUDA_ERR_CHECK(cudaMemset(pdevice_error, 0, sizeof(bool)));
		return;
	}
	
	// Transfer asyncio buffer contents.
	static char* pdevice_buffer = NULL;
	if (!pdevice_buffer)
		CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&pdevice_buffer, asyncio_buffer));
	vector<char> vhost_buffer;
	vhost_buffer.resize(host_length);
	char* host_buffer = &vhost_buffer[0];
	CUDA_ERR_CHECK(cudaMemcpy(host_buffer, pdevice_buffer, host_length,
		cudaMemcpyDeviceToHost));

	for (int offset = 0; offset < host_length; )
#else
	// Do nothing, if error status is true.
	if (asyncio_error)
	{
		asyncio_error = false;
		return;
	}

	for (int offset = 0; offset < asyncio_buffer_length; )
#endif

	{
#ifdef __CUDACC__
		transaction_t* t = (transaction_t*)(host_buffer + offset);
#else
		transaction_t* t = (transaction_t*)(asyncio_buffer + offset);
#endif
		offset += sizeof(transaction_t);
		t->offset = offset;
#ifdef __CUDACC__
		t->buffer = host_buffer;
#else
		t->buffer = asyncio_buffer;
#endif
		inside_hook_write = true;
		callback = st_write_callback;
		transaction = t;

		if ((t->format == ASYNCIO_UNFORMATTED) && (t->unit == ASYNCIO_STDOUT))
		{
			int get_st_parameter_val = setjmp(get_st_parameter_jmp);
			if (!get_st_parameter_val) asyncio_hook_write_default_unformatted();
		}
		else if ((t->format != ASYNCIO_UNFORMATTED) && (t->unit == ASYNCIO_STDOUT))
		{
			char* format = get_format(t->func, t->format);
			int get_st_parameter_val = setjmp(get_st_parameter_jmp);
			if (!get_st_parameter_val) asyncio_hook_write_default_formatted(strlen(format), format);
		}
		else if ((t->format == ASYNCIO_UNFORMATTED) && (t->unit != ASYNCIO_STDOUT))
		{
			int get_st_parameter_val = setjmp(get_st_parameter_jmp);
			if (!get_st_parameter_val) asyncio_hook_write_unit_unformatted(t->unit);
		}
		else
		{
			char* format = get_format(t->func, t->format);
			int get_st_parameter_val = setjmp(get_st_parameter_jmp);
			if (!get_st_parameter_val) asyncio_hook_write_unit_formatted(t->unit, strlen(format), format);
		}

		inside_hook_write = false;

		offset = t->offset;
	}

#ifdef __CUDACC__
	// Reset device pointer to 0, length to 0.
	static char* pdevice_pbuffer = NULL;
	if (!pdevice_pbuffer)
		CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&pdevice_pbuffer, asyncio_pbuffer));
	CUDA_ERR_CHECK(cudaMemset(pdevice_pbuffer, 0, sizeof(char*)));
	CUDA_ERR_CHECK(cudaMemset(pdevice_length, 0, sizeof(size_t)));
#else	
	// Reset device pointer to 0, length to 0.
	asyncio_pbuffer = NULL;
	asyncio_buffer_length = 0;
#endif
}

