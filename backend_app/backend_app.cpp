#include <windows.h>
#include <iostream>

int main()
{
	auto backend_dll = LoadLibrary(L"backend.dll");

	if (!backend_dll) {
		printf("Could not load backend dll!\n");
		return 1;
	}

	auto dll_main = (int (*)(void))GetProcAddress(backend_dll, "main");

	if (!dll_main) {
		printf("Could not get address of dll main function!\n");
		return 1;
	}

	dll_main();
}
