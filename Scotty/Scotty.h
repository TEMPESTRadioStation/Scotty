/* Scotty main header file
   It comprise main constants and variables */



// Delcare transmission constants and variables



// Default setup
// NOTE: When selecting a setup, one also need to select the setup at rs.h file
const double MinBitTime = 14; // microseconds
const int CodecCompression = 2; // Bits per sample: 2 to 5 bits
const int PacketsPerSecond = 32;
const int ReservedBytesPerSecond = 16;
const int PacketPerUnsignedCharThreshold = 32 * 7 - 1;

// Best bitrate setup
// NOTE: When selecting a setup, one also need to select the setup at rs.h file
//const double MinBitTime = 8; // microseconds
//const int CodecCompression = 2; // Bits per sample: 2 to 5 bits
//const int PacketsPerSecond = 32;
//const int ReservedBytesPerSecond = 16;
//const int PacketPerUnsignedCharThreshold = 32 * 7 - 1;

// Best audio quality setup
// This setup was tested on the laptop and on the desktop computer while the monitor was OFF (following power saving plan)
// NOTE: When selecting a setup, one also need to select the setup at rs.h file
//const double MinBitTime = 8; // microseconds
//const int CodecCompression = 3; // Bits per sample: 2 to 5 bits
//const int PacketsPerSecond = 48;
//const int ReservedBytesPerSecond = 24;
//const int PacketPerUnsignedCharThreshold = 48 * 5 - 1;



// Tx contants

/* MinBitTime is the minimal part of a symbol.
Transmitted symbol is defined by the length of a transmission.
Symbol length = ZeroOffset + MinBitTime * Data value.
Data value depends on the number of symbols per byte(of number of bits per symbol).
If 2 symbols per byte are selected, then each symbols represents 4 bits(Data value = 0 to 15).
If 4 symbols per byte are selected, then each symbols represents 2 bits(Data value = 0 to 3).*/

// Define BitTime constants. This defines minimal symbol length.
//const double MinBitTime = 22; // microseconds
//const double MinBitTime = 20; // microseconds
//const double MinBitTime = 14; // microseconds
//const double MinBitTime = 10; // microseconds
//const double MinBitTime = 8; // microseconds

// Define zero bit offest.
const int ZeroOffset = 1; // Zero nibble bit time offset

// Define packet length
const int PacketLength = 63 + 1; // In bytes, not including headers and footer
								 // A 63 bytes packet is good for complete voice messages
								 // with G.726 2,3,4 bits compression per symbol
								 // The plus 1 is for the packet counter

// Define number of symbols per data byte
//const int SymbolsPerByte = 4;
const int SymbolsPerByte = 2;

// Use forward error correction flag
bool useFEC = true;

// Define packet header bytes
const int TxHeader1 = 0x22;
const int TxHeader2 = 0x3C;
const int TxHeader3 = 0x3C;
const int TxHeader4 = 0x22;

// Set initial memory block size for memory data transfers
int MemoryBlockSize = 512 * 1024 * 1024;
const int MemoryBitsAllocation = 16 + ZeroOffset + 1;
const byte MemorySet0 = 0x00;
const byte MemorySet1 = 0x0F;
const byte MemorySet2 = 0x33;
byte *device_array_source, *device_array_dest;

// Tx thread end parameters
bool InTx = false;
bool EndTx = false;
const int EndTxTimeout = 20000; // [milliseconds]



// Audio processing variables

// Define audio sampling rate
const float CodecSamplingRate = 8000; // [Hz]

// Define G.726 number of bits per sample
//const int   CodecCompression = 2; // Bits per sample: 2 to 5 bits
//const int   CodecCompression = 3; // Bits per sample: 2 to 5 bits

// Define anti aliasing filter frequency
const float LowPassFilterFrequency = 4000; // [Hz]

// Define the number of packet that will be transmitted per 1 second.
// The goal if these calculation is to transmit 8000 samples per second.

// For CodecCompression = 2:
// PacketsPerSecond * PacketLength = 32 * 64 = 2048 bytes per second = 16384 bits per second
// The first byte in every packet is dedicated to a packet counter, so:
// Transmitted audio bits should be = PacketsPerSecondPacket * (PacketLength - 1) = 32 * 63 = 2016 audio bytes per second = 16128 bits per second
// To reduce the transmitted voice bits to exactly 16000, 16 bytes need to be reserved at the end of each packet cycle.
// This results with:
// Transmitted audio bits should be = PacketsPerSecondPacket * (PacketLength - 1) - 16 = 32 * 63 - 16 = 2000 audio bytes per second = 16000 audio bits per second 
// Total transmitter bits per second = PacketsPerSecond * PacketLength = 32 * (63 + 1 + 4 + 2) = 2240 bytes per second = 17920 bits per second

//const int PacketsPerSecond = 32;
//const int ReservedBytesPerSecond = 16;
//const int PacketPerUnsignedCharThreshold = 32 * 7 - 1;

// For CodecCompression = 3:
// PacketsPerSecond * PacketLength = 48 * 64 = 3072 bytes per second = 24576 bits per second
// The first byte in every packet is dedicated to a packet counter, so:
// Transmitted audio bits should be = PacketsPerSecondPacket * (PacketLength - 1) = 48 * 63 = 3024 audio bytes per second = 24192 bits per second
// To reduce the transmitted voice bits to exactly 24000, 24 bytes need to be reserved at the end of each packet cycle.
// This results with:
// Transmitted audio bits should be = PacketsPerSecondPacket * (PacketLength - 1) - 24 = 48 * 63 - 24 = 3000 audio bytes per second = 24000 audio bits per second 
// Total transmitter bits per second = PacketsPerSecond * PacketLength = 48 * (63 + 1 + 4 + 2) = 3360 bytes per second = 26880 bits per second

//const int PacketsPerSecond = 48;
//const int ReservedBytesPerSecond = 24;
//const int PacketPerUnsignedCharThreshold = 48 * 5 - 1;




// Declare NvApi declaration and structs

#define MAKE_NVAPI_VERSION(type, version) ((unsigned int)(sizeof(type) | ((version) << 16)))

struct NV_GPU_PERF_PSTATES20_PARAM_DELTA
{
	int value;

	struct
	{
		int min;
		int max;
	} valueRange;
};

struct NV_GPU_PSTATE20_BASE_VOLTAGE_ENTRY
{
	unsigned int domainId;
	unsigned int editable : 1;
	unsigned int reserved : 31;
	unsigned int voltageUV;
	NV_GPU_PERF_PSTATES20_PARAM_DELTA voltageDeltaUV;
};

struct NV_GPU_PSTATE20_CLOCK_ENTRY
{
	unsigned int domainId;
	unsigned int typeId;
	unsigned int editable : 1;
	unsigned int reserved : 31;
	NV_GPU_PERF_PSTATES20_PARAM_DELTA frequencyDeltaKHz;

	union
	{
		struct
		{
			unsigned int frequencyKHz;
		} single;

		struct
		{
			unsigned int minFrequencyKHz;
			unsigned int maxFrequencyKHz;
			unsigned int domainId;
			unsigned int minVoltageUV;
			unsigned int maxVoltageUV;
		} range;
	} data;
};

struct NV_GPU_PERF_PSTATES20_INFO
{
	unsigned int version;
	unsigned int editable : 1;
	unsigned int reserved : 31;
	unsigned int numPStates;
	unsigned int numClocks;
	unsigned int numBaseVoltages;

	struct
	{
		unsigned int pStateId;
		unsigned int editable : 1;
		unsigned int reserved : 31;
		NV_GPU_PSTATE20_CLOCK_ENTRY clocks[8];
		NV_GPU_PSTATE20_BASE_VOLTAGE_ENTRY baseVoltages[4];
	} pStates[16];

	struct
	{
		unsigned int numVoltages;
		NV_GPU_PSTATE20_BASE_VOLTAGE_ENTRY voltages[4];
	} overVoltage;
};

typedef struct _NV_CLOCKS_INFO
{
	unsigned int version;
	unsigned int clocks[288];
} NV_CLOCKS_INFO;

#define NV_CLOCKS_INFO_VER  MAKE_NVAPI_VERSION(NV_CLOCKS_INFO, 2)

struct NV_GPU_CLOCK_FREQUENCIES_V2
{
	unsigned long   version;
	unsigned long   ClockType : 4;
	unsigned long   reserved : 20;
	unsigned long   reserved1 : 8;
	struct
	{
		unsigned long bIsPresent : 1;
		unsigned long reserved : 31;
		unsigned long frequency;
	}domain[32];
};
typedef struct NV_GPU_CLOCK_FREQUENCIES_V2 NV_GPU_CLOCK_FREQUENCIES;

#define NV_GPU_CLOCK_FREQUENCIES_VER_1    MAKE_NVAPI_VERSION(NV_GPU_CLOCK_FREQUENCIES_V1,1)
#define NV_GPU_CLOCK_FREQUENCIES_VER_2    MAKE_NVAPI_VERSION(NV_GPU_CLOCK_FREQUENCIES_V2,2)
#define NV_GPU_CLOCK_FREQUENCIES_VER_3    MAKE_NVAPI_VERSION(NV_GPU_CLOCK_FREQUENCIES_V2,3)
#define NV_GPU_CLOCK_FREQUENCIES_VER	  NV_GPU_CLOCK_FREQUENCIES_VER_3

typedef enum _NV_GPU_PUBLIC_CLOCK_ID
{
	NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS = 0,
	NVAPI_GPU_PUBLIC_CLOCK_MEMORY = 4,
	NVAPI_GPU_PUBLIC_CLOCK_PROCESSOR = 7,
	NVAPI_GPU_PUBLIC_CLOCK_VIDEO = 8,
	NVAPI_GPU_PUBLIC_CLOCK_UNDEFINED = 32,
} NV_GPU_PUBLIC_CLOCK_ID;

typedef enum
{
	NV_GPU_CLOCK_FREQUENCIES_CURRENT_FREQ = 0,
	NV_GPU_CLOCK_FREQUENCIES_BASE_CLOCK = 1,
	NV_GPU_CLOCK_FREQUENCIES_BOOST_CLOCK = 2,
	NV_GPU_CLOCK_FREQUENCIES_CLOCK_TYPE_NUM = 3
}   NV_GPU_CLOCK_FREQUENCIES_CLOCK_TYPE;

// Declare NvApi functions

void* (__cdecl *NvAPI_QueryInterface)(unsigned int offset) = 0;
int(__cdecl *NvAPI_Initialize)() = 0;
int(__cdecl *NvAPI_Unload)() = 0;
int(__cdecl *NvAPI_EnumPhysicalGPUs)(void **gpuHandles, unsigned int *gpuCount) = 0;
int(__cdecl *NvAPI_GPU_GetBusId)(void *gpuHandle, unsigned int *busId) = 0;
int(__cdecl* NvAPI_GPU_GetFullName)(void* gpuHandle, char NvAPI_ShortString[64]) = 0;
int(__cdecl *NvAPI_GPU_GetAllClocks)(void *gpuHandle, NV_CLOCKS_INFO *pClocksInfo) = 0;
int(__cdecl *NvAPI_GPU_GetAllClockFrequencies)(void *gpuHandle, NV_GPU_CLOCK_FREQUENCIES *pClkFreqs) = 0;
int(__cdecl *NvAPI_GPU_GetRamType)(void *gpuHandle, unsigned int *GPUMemoryType) = 0;
int(__cdecl *NvAPI_GPU_GetPstates20)(void *gpuHandle, NV_GPU_PERF_PSTATES20_INFO *pStatesInfo) = 0;
int(__cdecl *NvAPI_GPU_SetPstates20)(void *gpuHandle, NV_GPU_PERF_PSTATES20_INFO *pStatesInfo) = 0;



// WAV file properties (used when saving debug data)

//Define WAV file header
struct wav_header_t
{
	char chunkID[4];
	unsigned long chunkSize;
	char format[4];
	char subchunk1ID[4];
	unsigned long subchunk1Size;
	unsigned short audioFormat;
	unsigned short numChannels;
	unsigned long sampleRate;
	unsigned long byteRate;
	unsigned short blockAlign;
	unsigned short bitsPerSample;
};

//Define WAV file chunks

//Chunks
struct chunk_t
{
	char ID[4];
	unsigned long size;
};
