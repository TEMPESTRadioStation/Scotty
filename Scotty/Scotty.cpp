/* Scotty

Scotty is a demonstartion software. It demonstrates how audio may be transmitted from a host computer by controlling 
GPU memory data transfers. Each memory transfer generates electromagnetic waves. By controlling the memory data
tranfers, the software transmitts the data from the computer.
Scotty was developed on an Asus G731GU-BI7N9 laptop with an nVIDIA GeForce GTX 1660 Ti GPU and
tested on a desktop computer which comprise an ASUS TUF GTX 1650 SUPER GAMING 4GB GDDR6 card.
It uses nVIDIA CUDA to perform the memory transfers.

Scotty performs the following tasks:
1. It learns how long it takes the host computer to transfer a given amount of data.
2. It calculates the amount of data required to transmit a single bit.
   A single bit is the minimal part of a symbol.
   Transmitted symbol is defined by the length of a transmission.
   Symbol length = ZeroOffset + MinBitTime * Data value.
   Data value depends on the number of symbols per byte (of number of bits per symbol).
   If 2 symbols per byte are selected, then each symbols represents 4 bits (Data value = 0 to 15).
   If 4 symbols per byte are selected, then each symbols represents 2 bits (Data value = 0 to 3).
3. It enables the user to control the memory clock.
   The data is transmitted at this frequency, so by controlling the memory clock the user can transmit the data in a inactive
   RF band.
   Note: The data is also transmitted in other frequencies, such as the bytes data rate frequency.
4. It enables the user to select the WAV file to be transmitted.
5. When a WAV file is transmitted, the software performs the following tasks:
   5.1. It reads the file.
   5.2. It filters the audio with a 4000Hz filter and downsample it to 8000 samples per second.
   5.3. It uses a G.726 endocder to reduce the amount of bits per sample.
   5.4. It gather the data bits in data packets, which include 16 bit checksum.
   5.5. It applies a Reed-Solomon forward error correction (FEC) algorithem on the data.
   5.6. For each data packet it transmits, 4 header bytes, FEC parity bytes and data bytes.
Audio data is transmitted at a rate of 8000 samples per second.
The data is transmitted in OOK.

Scotty was released by Paz Hameiri on 2021.

It includes a slightly modified version of Simon Rockliff' Reed-Solomon encoding-decoding code.
It also includes a G.726 code, provided by Sun Microsystems, Inc.

Permission to use, copy, modify, and/or distribute the software that were written by Paz Hameiri for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
OF THIS SOFTWARE.

*/

#include "Windows.h"
#include "ScottyGUI.h"

// Define NvApi GPU handles array
void *NvApiGpuHandles[128] = { 0 };

// Load NvApi DLL and setup the functions that will be used.
// DLL selection is done by following the computer's windows version (64 bits or 32 bits)
int NvApiLoad()
{
	int result = -1;

	HMODULE nvapi = 0;

#if defined _WIN64
	char nvapiDllName[] = "nvapi64.dll";
#else
	char nvapiDllName[] = "nvapi.dll";
#endif

	nvapi = LoadLibraryA(nvapiDllName);

	result = !(nvapi != 0);

	if (result == 0)
	{
		NvAPI_QueryInterface = (void* (__cdecl*)(unsigned int)) GetProcAddress(nvapi, "nvapi_QueryInterface");

		result = !(NvAPI_QueryInterface != 0);

		if (result == 0)
		{
			NvAPI_Initialize = (int(__cdecl *)()) NvAPI_QueryInterface(0x0150E828);
			NvAPI_Unload = (int(__cdecl*)()) NvAPI_QueryInterface(0xD22BDD7E);
			NvAPI_EnumPhysicalGPUs = (int(__cdecl*)(void**, unsigned int*)) NvAPI_QueryInterface(0xE5AC921F);
			NvAPI_GPU_SetPstates20 = (int(__cdecl*)(void*, NV_GPU_PERF_PSTATES20_INFO*)) NvAPI_QueryInterface(0x0F4DAE6B);
			NvAPI_GPU_GetBusId = (int(__cdecl *)(void*, unsigned int*)) NvAPI_QueryInterface(0x1BE0B8E5);
			NvAPI_GPU_GetFullName = (int(__cdecl*)(void*, char NvAPI_ShortString[64])) NvAPI_QueryInterface(0xCEEE8E9F);
			NvAPI_GPU_GetAllClocks = (int(__cdecl*)(void*, NV_CLOCKS_INFO*)) NvAPI_QueryInterface(0x1BD69F49);
			NvAPI_GPU_GetAllClockFrequencies = (int(__cdecl*)(void*, NV_GPU_CLOCK_FREQUENCIES*)) NvAPI_QueryInterface(0xDCB616C3);
			NvAPI_GPU_GetRamType = (int(__cdecl*)(void*, unsigned int*)) NvAPI_QueryInterface(0x57F7CAAC);
			NvAPI_GPU_GetPstates20 = (int(__cdecl*)(void*, NV_GPU_PERF_PSTATES20_INFO*)) NvAPI_QueryInterface(0x6FF81213);
		}
	}

	return result;
}


// Initialise NvApi
int NvApiInit()
{
	int result = -1;

	if (NvAPI_Initialize)
	{
		result = NvAPI_Initialize();
	}

	return result;
}

// Free NvApi
int NvApiFree()
{
	int result = -1;

	if (NvAPI_Unload)
	{
		result = NvAPI_Unload();
	}

	return result;
}

using namespace System;
using namespace System::Windows::Forms;

// Constants
const int MaxHandeledGPUs = 4;

// Declare global variables
void* Handles[64] = { 0 }; // Handles to the GPUs
unsigned int NumberOfPGUs = 0; // Hold the number of GPUs out of which the user can select - 1 to MaxHandeledGPUs.
unsigned int BusId[MaxHandeledGPUs] = { 0 }; // Hold the bus ID of each GPU
char GPUsFullName[MaxHandeledGPUs * 64]; // Sets of 64 characters, for up to 4 GPUs names

// Perform a memory test to evalute the time it takes to transmit a certain amount of data and calculate the data per time ratio.
double MemorySpeedTest(int TestMemoryBlockSize)
{
	// Allocate memory at the device
	cudaMalloc((void**)&device_array_source, TestMemoryBlockSize);
	cudaMalloc((void**)&device_array_dest, TestMemoryBlockSize);

	// Set source memory to MemorySet
	cudaMemset(device_array_source, MemorySet0, TestMemoryBlockSize);
	cudaDeviceSynchronize();

	// Declare averaging parameters
	double AverageRatio = 0;
	const int AveragingCycles = 256;

	// Run speed test to find the minimal byte count per microsecond
	for (int i = 0; i < AveragingCycles; i++)
	{
		// Reduce block size every 32 counts
		int CurrentMemoryBlockSize = TestMemoryBlockSize;

		// Get present time
		high_resolution_clock::time_point t0 = high_resolution_clock::now();

		// Perform a memory data transfer
		cudaMemcpy(device_array_dest, device_array_source, CurrentMemoryBlockSize, cudaMemcpyDeviceToDevice);
		// Synchronise between the host CPU and the GPU. It triggers the GPU to perfrom the memory transfer.
		cudaDeviceSynchronize();

		// Calculate the time diff.
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		duration<double, std::nano> time_span = t1 - t0;
		double Time_interval = Convert::ToDouble(time_span.count())/1000;

		// Accumulate bytes per microsecond
		AverageRatio = AverageRatio + (double)CurrentMemoryBlockSize / Time_interval;
	}

	// Calculate average bytes per microsecond
	AverageRatio = AverageRatio / AveragingCycles;

	// Free device memory
	cudaFree(device_array_source);
	cudaFree(device_array_dest);
	cudaDeviceSynchronize();
	return(AverageRatio);
}

[STAThread]
// C++ main
void Main() {

	// Check if the GPU driver is available and have the required functions
	if (NvApiLoad() == 0)
	{
		// Initialize the process and check if the initialization process went OK
		if (NvApiInit() == 0)
		{
			int result = -1;

			// Check if a card is available
			if ((NvAPI_EnumPhysicalGPUs) && (NvAPI_GPU_GetBusId))
			{
				unsigned int GPUs_count = 0;

				// Get the GPUs count and handles
				result = NvAPI_EnumPhysicalGPUs(Handles, &GPUs_count);

				// Check if the last action got valid results
				if (result == 0)
				{
					// Store the number of handeled GPUs to MaxHandeledGPUs. Limit the number if needed.
					if (GPUs_count > MaxHandeledGPUs) NumberOfPGUs = MaxHandeledGPUs;
					else NumberOfPGUs = GPUs_count;

					// Gather GPUs information
					for (unsigned int i = 0; i < NumberOfPGUs; ++i)
					{
						result = 0;

						// Get BusId
						result = NvAPI_GPU_GetBusId(Handles[i], &BusId[i]);
						
						// Check if the last action got valid results
						if (result == 0)
						{
							// Store handles per BusId
							NvApiGpuHandles[BusId[i]] = Handles[i];

							char FullName[64];

							result = 0;

							// Store the cards' full names in a buffer
							result = NvAPI_GPU_GetFullName(Handles[i], FullName);

							if (result == 0)
							{
								char *pd = GPUsFullName + 64 * i;
								strncpy_s(pd, 64, FullName, 64);
							}
						}
					}
				}
			}
		}
	}

	// If cards are available - run the rest of the software
	if (NumberOfPGUs > 0)
	{
		// Perform memory speed test
		double Average_Ratio = MemorySpeedTest(MemoryBlockSize);

		// Calculte MemoryBlockSize adapted to minimal bit time
		MemoryBlockSize = (int)(MinBitTime * Average_Ratio);

		// Allocate memory at the device
		cudaMalloc((void**)&device_array_source, MemoryBitsAllocation*MemoryBlockSize);
		cudaMalloc((void**)&device_array_dest, MemoryBitsAllocation*MemoryBlockSize);

		// Set source memory to MemorySet0
		cudaMemset(device_array_source, MemorySet0, MemoryBitsAllocation*MemoryBlockSize);
		cudaDeviceSynchronize();

		// Launch the form
		Application::EnableVisualStyles();
		Application::SetCompatibleTextRenderingDefault(false);
		Scotty::MyForm form;
		Application::Run(%form);

		// End Tx thread
		EndTx = true;

		// Wait for Tx thread to end
		int ElapsedTime = 0;
		if (InTx) while (EndTx & (ElapsedTime < EndTxTimeout))
		{
			ElapsedTime += 200;
			Sleep(200);
		}

		// Free NvApi resources
		NvApiFree();

		// Free device memory
		cudaFree(device_array_source);
		cudaFree(device_array_dest);
	}
}