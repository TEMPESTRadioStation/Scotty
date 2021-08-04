// Scotty form code

// Declare global constants
extern const int MaxHandeledGPUs;

// Declare global variables
extern void* Handles[];
extern unsigned int NumberOfPGUs;
extern unsigned int BusId[];
extern void *NvApiGpuHandles[];
extern char GPUsFullName[];

// Declare FEC global variables
extern int FECdata[];
extern int bb[];

#include <string>
#include "Scotty.h"
#include <chrono>
#include "cuda_runtime.h"
#include "g72x.h"
#include "rs.h"

using namespace std::chrono;
using namespace System;
using namespace System::Threading;
using namespace System::IO;
using namespace System::Runtime::InteropServices;

#pragma once

namespace Scotty {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for MyForm
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			// Initialize form components
			InitializeComponent();

			// Place default WAV file name in the file name box
			InitWAVFileName();

			// Initialize the values of the "Divider" drop list and select the default value
			InitDivider();

			// Initialize the values of the "Data" drop list and select the default value
			InitData();

			// Initialize the values of the "GPUs list" list and select the default value
			InitTable();

			// Generate the Galois Field GF(2**mm) for the Reed-Solomon encoder
			generate_gf();

			// Compute the generator polynomial for this RS code for the Reed-Solomon encoder
			gen_poly();

			// Set all bytes in FEC data buffer to 0
			for (int i = 0; i < kk; i++) FECdata[i] = 0;

			// Initiate an internal tick loop, to perform cyclic actions within the form
			InitInnerTimer(500);
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}

	private: System::Windows::Forms::ListBox^  GPUsList;
	private: System::Windows::Forms::Label^  GPUs_list_label;
	private: System::Windows::Forms::Label^  Memory_clock_label;
	private: System::Windows::Forms::Label^  MCMhz_label;
	private: System::Windows::Forms::ComboBox^  Divider_comboBox;
	private: System::Windows::Forms::Label^  Divider_label;
	private: System::Windows::Forms::TextBox^  Memory_clock_textBox;
	private: System::Windows::Forms::Label^  MBCMhz_label;
	private: System::Windows::Forms::Label^  Memory_base_label;
	private: System::Windows::Forms::TextBox^  Memory_base_textBox;
	private: System::Windows::Forms::NumericUpDown^  BaseShiftnumeric;
	private: System::Windows::Forms::Label^  Base_clock_label;
	private: System::Windows::Forms::Label^  BCMHz_label;
	private: System::Windows::Forms::Label^  CFMHz_label;
	private: System::Windows::Forms::Label^  Center_freq_label;
	private: System::Windows::Forms::TextBox^  Center_Freq_textBox;
	private: System::Windows::Forms::CheckBox^  Shift_checkBox;
	private: System::Windows::Forms::Label^  MFMhz_label;
	private: System::Windows::Forms::Label^  MeasuredTx_freq_label;

	private: System::Windows::Forms::TextBox^  Measured_freq_textBox;
	private: System::Windows::Forms::CheckBox^  Tx_test_stream_checkBox;
	private: System::Windows::Forms::Label^  kbpslabel;
	private: System::Windows::Forms::TextBox^  kbps_textBox;





	private: System::Windows::Forms::TextBox^  Tx_Progress_textBox;
	private: System::Windows::Forms::Label^  Tx_Precentage_label;
	private: System::Windows::Forms::CheckBox^  Tx_wav_checkBox;
	private: System::Windows::Forms::Label^  WAVFileName_label;
	private: System::Windows::Forms::TextBox^  WAV_file_name_textBox;
	private: System::Windows::Forms::Label^  TxProp_label;
	private: System::Windows::Forms::Label^  Data_label;
	private: System::Windows::Forms::ComboBox^  Data_comboBox;
	private: System::Windows::Forms::Label^  RawTx_label;
	private: System::Windows::Forms::Label^  rawkbpslabel;
	private: System::Windows::Forms::TextBox^  raw_kbps_textBox;
	private: System::Windows::Forms::Label^  data_transmitted_label;

	protected:

	protected:

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>

		// Initialize form components

		void InitializeComponent(void)
		{
			this->GPUsList = (gcnew System::Windows::Forms::ListBox());
			this->GPUs_list_label = (gcnew System::Windows::Forms::Label());
			this->Memory_clock_label = (gcnew System::Windows::Forms::Label());
			this->MCMhz_label = (gcnew System::Windows::Forms::Label());
			this->Divider_comboBox = (gcnew System::Windows::Forms::ComboBox());
			this->Divider_label = (gcnew System::Windows::Forms::Label());
			this->Memory_clock_textBox = (gcnew System::Windows::Forms::TextBox());
			this->MBCMhz_label = (gcnew System::Windows::Forms::Label());
			this->Memory_base_label = (gcnew System::Windows::Forms::Label());
			this->Memory_base_textBox = (gcnew System::Windows::Forms::TextBox());
			this->BaseShiftnumeric = (gcnew System::Windows::Forms::NumericUpDown());
			this->Base_clock_label = (gcnew System::Windows::Forms::Label());
			this->BCMHz_label = (gcnew System::Windows::Forms::Label());
			this->CFMHz_label = (gcnew System::Windows::Forms::Label());
			this->Center_freq_label = (gcnew System::Windows::Forms::Label());
			this->Center_Freq_textBox = (gcnew System::Windows::Forms::TextBox());
			this->Shift_checkBox = (gcnew System::Windows::Forms::CheckBox());
			this->MFMhz_label = (gcnew System::Windows::Forms::Label());
			this->MeasuredTx_freq_label = (gcnew System::Windows::Forms::Label());
			this->Measured_freq_textBox = (gcnew System::Windows::Forms::TextBox());
			this->Tx_test_stream_checkBox = (gcnew System::Windows::Forms::CheckBox());
			this->kbpslabel = (gcnew System::Windows::Forms::Label());
			this->kbps_textBox = (gcnew System::Windows::Forms::TextBox());
			this->Tx_Progress_textBox = (gcnew System::Windows::Forms::TextBox());
			this->Tx_Precentage_label = (gcnew System::Windows::Forms::Label());
			this->Tx_wav_checkBox = (gcnew System::Windows::Forms::CheckBox());
			this->WAVFileName_label = (gcnew System::Windows::Forms::Label());
			this->WAV_file_name_textBox = (gcnew System::Windows::Forms::TextBox());
			this->TxProp_label = (gcnew System::Windows::Forms::Label());
			this->Data_label = (gcnew System::Windows::Forms::Label());
			this->Data_comboBox = (gcnew System::Windows::Forms::ComboBox());
			this->RawTx_label = (gcnew System::Windows::Forms::Label());
			this->rawkbpslabel = (gcnew System::Windows::Forms::Label());
			this->raw_kbps_textBox = (gcnew System::Windows::Forms::TextBox());
			this->data_transmitted_label = (gcnew System::Windows::Forms::Label());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->BaseShiftnumeric))->BeginInit();
			this->SuspendLayout();
			// 
			// GPUsList
			// 
			this->GPUsList->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->GPUsList->FormattingEnabled = true;
			this->GPUsList->ItemHeight = 23;
			this->GPUsList->Location = System::Drawing::Point(12, 44);
			this->GPUsList->Name = L"GPUsList";
			this->GPUsList->Size = System::Drawing::Size(353, 96);
			this->GPUsList->TabIndex = 2;
			this->GPUsList->SelectedIndexChanged += gcnew System::EventHandler(this, &MyForm::GPUsList_SelectedIndexChanged);
			// 
			// GPUs_list_label
			// 
			this->GPUs_list_label->AutoSize = true;
			this->GPUs_list_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->GPUs_list_label->Location = System::Drawing::Point(12, 9);
			this->GPUs_list_label->Name = L"GPUs_list_label";
			this->GPUs_list_label->Size = System::Drawing::Size(98, 24);
			this->GPUs_list_label->TabIndex = 3;
			this->GPUs_list_label->Text = L"GPUs list";
			// 
			// Memory_clock_label
			// 
			this->Memory_clock_label->AutoSize = true;
			this->Memory_clock_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Memory_clock_label->Location = System::Drawing::Point(394, 9);
			this->Memory_clock_label->Name = L"Memory_clock_label";
			this->Memory_clock_label->Size = System::Drawing::Size(142, 24);
			this->Memory_clock_label->TabIndex = 6;
			this->Memory_clock_label->Text = L"Memory clock";
			// 
			// MCMhz_label
			// 
			this->MCMhz_label->AutoSize = true;
			this->MCMhz_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->MCMhz_label->Location = System::Drawing::Point(542, 47);
			this->MCMhz_label->Name = L"MCMhz_label";
			this->MCMhz_label->Size = System::Drawing::Size(46, 23);
			this->MCMhz_label->TabIndex = 7;
			this->MCMhz_label->Text = L"Mhz";
			// 
			// Divider_comboBox
			// 
			this->Divider_comboBox->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Divider_comboBox->FormattingEnabled = true;
			this->Divider_comboBox->Location = System::Drawing::Point(659, 44);
			this->Divider_comboBox->Name = L"Divider_comboBox";
			this->Divider_comboBox->Size = System::Drawing::Size(63, 31);
			this->Divider_comboBox->TabIndex = 9;
			this->Divider_comboBox->SelectedIndexChanged += gcnew System::EventHandler(this, &MyForm::Divider_comboBox_SelectedIndexChanged);
			// 
			// Divider_label
			// 
			this->Divider_label->AutoSize = true;
			this->Divider_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Divider_label->Location = System::Drawing::Point(655, 9);
			this->Divider_label->Name = L"Divider_label";
			this->Divider_label->Size = System::Drawing::Size(76, 24);
			this->Divider_label->TabIndex = 10;
			this->Divider_label->Text = L"Divider";
			// 
			// Memory_clock_textBox
			// 
			this->Memory_clock_textBox->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Memory_clock_textBox->Location = System::Drawing::Point(398, 44);
			this->Memory_clock_textBox->Name = L"Memory_clock_textBox";
			this->Memory_clock_textBox->Size = System::Drawing::Size(138, 30);
			this->Memory_clock_textBox->TabIndex = 5;
			// 
			// MBCMhz_label
			// 
			this->MBCMhz_label->AutoSize = true;
			this->MBCMhz_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->MBCMhz_label->Location = System::Drawing::Point(1052, 47);
			this->MBCMhz_label->Name = L"MBCMhz_label";
			this->MBCMhz_label->Size = System::Drawing::Size(46, 23);
			this->MBCMhz_label->TabIndex = 13;
			this->MBCMhz_label->Text = L"Mhz";
			// 
			// Memory_base_label
			// 
			this->Memory_base_label->AutoSize = true;
			this->Memory_base_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Memory_base_label->Location = System::Drawing::Point(904, 9);
			this->Memory_base_label->Name = L"Memory_base_label";
			this->Memory_base_label->Size = System::Drawing::Size(193, 24);
			this->Memory_base_label->TabIndex = 12;
			this->Memory_base_label->Text = L"Memory base clock";
			// 
			// Memory_base_textBox
			// 
			this->Memory_base_textBox->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Memory_base_textBox->Location = System::Drawing::Point(908, 44);
			this->Memory_base_textBox->Name = L"Memory_base_textBox";
			this->Memory_base_textBox->Size = System::Drawing::Size(138, 30);
			this->Memory_base_textBox->TabIndex = 11;
			// 
			// BaseShiftnumeric
			// 
			this->BaseShiftnumeric->DecimalPlaces = 3;
			this->BaseShiftnumeric->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->BaseShiftnumeric->Location = System::Drawing::Point(398, 126);
			this->BaseShiftnumeric->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 250, 0, 0, 0 });
			this->BaseShiftnumeric->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 250, 0, 0, System::Int32::MinValue });
			this->BaseShiftnumeric->Name = L"BaseShiftnumeric";
			this->BaseShiftnumeric->Size = System::Drawing::Size(136, 30);
			this->BaseShiftnumeric->TabIndex = 14;
			this->BaseShiftnumeric->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			this->BaseShiftnumeric->ValueChanged += gcnew System::EventHandler(this, &MyForm::BaseShiftnumeric_ValueChanged);
			// 
			// Base_clock_label
			// 
			this->Base_clock_label->AutoSize = true;
			this->Base_clock_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Base_clock_label->Location = System::Drawing::Point(394, 90);
			this->Base_clock_label->Name = L"Base_clock_label";
			this->Base_clock_label->Size = System::Drawing::Size(161, 24);
			this->Base_clock_label->TabIndex = 15;
			this->Base_clock_label->Text = L"Base clock shift";
			// 
			// BCMHz_label
			// 
			this->BCMHz_label->AutoSize = true;
			this->BCMHz_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->BCMHz_label->Location = System::Drawing::Point(542, 128);
			this->BCMHz_label->Name = L"BCMHz_label";
			this->BCMHz_label->Size = System::Drawing::Size(46, 23);
			this->BCMHz_label->TabIndex = 16;
			this->BCMHz_label->Text = L"Mhz";
			// 
			// CFMHz_label
			// 
			this->CFMHz_label->AutoSize = true;
			this->CFMHz_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->CFMHz_label->Location = System::Drawing::Point(1052, 128);
			this->CFMHz_label->Name = L"CFMHz_label";
			this->CFMHz_label->Size = System::Drawing::Size(46, 23);
			this->CFMHz_label->TabIndex = 19;
			this->CFMHz_label->Text = L"Mhz";
			// 
			// Center_freq_label
			// 
			this->Center_freq_label->AutoSize = true;
			this->Center_freq_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Center_freq_label->Location = System::Drawing::Point(904, 90);
			this->Center_freq_label->Name = L"Center_freq_label";
			this->Center_freq_label->Size = System::Drawing::Size(174, 24);
			this->Center_freq_label->TabIndex = 18;
			this->Center_freq_label->Text = L"Center frequency";
			// 
			// Center_Freq_textBox
			// 
			this->Center_Freq_textBox->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Center_Freq_textBox->Location = System::Drawing::Point(908, 125);
			this->Center_Freq_textBox->Name = L"Center_Freq_textBox";
			this->Center_Freq_textBox->Size = System::Drawing::Size(138, 30);
			this->Center_Freq_textBox->TabIndex = 17;
			// 
			// Shift_checkBox
			// 
			this->Shift_checkBox->AutoSize = true;
			this->Shift_checkBox->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Shift_checkBox->Location = System::Drawing::Point(654, 129);
			this->Shift_checkBox->Name = L"Shift_checkBox";
			this->Shift_checkBox->Size = System::Drawing::Size(170, 27);
			this->Shift_checkBox->TabIndex = 21;
			this->Shift_checkBox->Text = L"Shift frerquency";
			this->Shift_checkBox->UseVisualStyleBackColor = true;
			this->Shift_checkBox->CheckedChanged += gcnew System::EventHandler(this, &MyForm::Shift_checkBox_CheckedChanged);
			// 
			// MFMhz_label
			// 
			this->MFMhz_label->AutoSize = true;
			this->MFMhz_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->MFMhz_label->Location = System::Drawing::Point(1052, 211);
			this->MFMhz_label->Name = L"MFMhz_label";
			this->MFMhz_label->Size = System::Drawing::Size(46, 23);
			this->MFMhz_label->TabIndex = 24;
			this->MFMhz_label->Text = L"Mhz";
			// 
			// MeasuredTx_freq_label
			// 
			this->MeasuredTx_freq_label->AutoSize = true;
			this->MeasuredTx_freq_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->MeasuredTx_freq_label->Location = System::Drawing::Point(904, 173);
			this->MeasuredTx_freq_label->Name = L"MeasuredTx_freq_label";
			this->MeasuredTx_freq_label->Size = System::Drawing::Size(234, 24);
			this->MeasuredTx_freq_label->TabIndex = 23;
			this->MeasuredTx_freq_label->Text = L"Measured Tx frequency";
			// 
			// Measured_freq_textBox
			// 
			this->Measured_freq_textBox->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Measured_freq_textBox->Location = System::Drawing::Point(908, 208);
			this->Measured_freq_textBox->Name = L"Measured_freq_textBox";
			this->Measured_freq_textBox->Size = System::Drawing::Size(138, 30);
			this->Measured_freq_textBox->TabIndex = 22;
			// 
			// Tx_test_stream_checkBox
			// 
			this->Tx_test_stream_checkBox->AutoSize = true;
			this->Tx_test_stream_checkBox->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Tx_test_stream_checkBox->Location = System::Drawing::Point(12, 167);
			this->Tx_test_stream_checkBox->Name = L"Tx_test_stream_checkBox";
			this->Tx_test_stream_checkBox->Size = System::Drawing::Size(159, 27);
			this->Tx_test_stream_checkBox->TabIndex = 25;
			this->Tx_test_stream_checkBox->Text = L"Tx test stream";
			this->Tx_test_stream_checkBox->UseVisualStyleBackColor = true;
			this->Tx_test_stream_checkBox->CheckedChanged += gcnew System::EventHandler(this, &MyForm::Tx_checkBox_CheckedChanged);
			// 
			// kbpslabel
			// 
			this->kbpslabel->AutoSize = true;
			this->kbpslabel->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->kbpslabel->Location = System::Drawing::Point(498, 294);
			this->kbpslabel->Name = L"kbpslabel";
			this->kbpslabel->Size = System::Drawing::Size(52, 23);
			this->kbpslabel->TabIndex = 28;
			this->kbpslabel->Text = L"kbps";
			// 
			// kbps_textBox
			// 
			this->kbps_textBox->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->kbps_textBox->Location = System::Drawing::Point(398, 291);
			this->kbps_textBox->Name = L"kbps_textBox";
			this->kbps_textBox->Size = System::Drawing::Size(90, 30);
			this->kbps_textBox->TabIndex = 26;
			// 
			// Tx_Progress_textBox
			// 
			this->Tx_Progress_textBox->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Tx_Progress_textBox->Location = System::Drawing::Point(654, 291);
			this->Tx_Progress_textBox->Name = L"Tx_Progress_textBox";
			this->Tx_Progress_textBox->Size = System::Drawing::Size(52, 30);
			this->Tx_Progress_textBox->TabIndex = 32;
			// 
			// Tx_Precentage_label
			// 
			this->Tx_Precentage_label->AutoSize = true;
			this->Tx_Precentage_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Tx_Precentage_label->Location = System::Drawing::Point(712, 294);
			this->Tx_Precentage_label->Name = L"Tx_Precentage_label";
			this->Tx_Precentage_label->Size = System::Drawing::Size(28, 23);
			this->Tx_Precentage_label->TabIndex = 33;
			this->Tx_Precentage_label->Text = L"%";
			// 
			// Tx_wav_checkBox
			// 
			this->Tx_wav_checkBox->AutoSize = true;
			this->Tx_wav_checkBox->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Tx_wav_checkBox->Location = System::Drawing::Point(12, 208);
			this->Tx_wav_checkBox->Name = L"Tx_wav_checkBox";
			this->Tx_wav_checkBox->Size = System::Drawing::Size(133, 27);
			this->Tx_wav_checkBox->TabIndex = 34;
			this->Tx_wav_checkBox->Text = L"Tx WAV file";
			this->Tx_wav_checkBox->UseVisualStyleBackColor = true;
			this->Tx_wav_checkBox->CheckedChanged += gcnew System::EventHandler(this, &MyForm::Tx_wav_checkBox_CheckedChanged);
			// 
			// WAVFileName_label
			// 
			this->WAVFileName_label->AutoSize = true;
			this->WAVFileName_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->WAVFileName_label->Location = System::Drawing::Point(12, 327);
			this->WAVFileName_label->Name = L"WAVFileName_label";
			this->WAVFileName_label->Size = System::Drawing::Size(144, 24);
			this->WAVFileName_label->TabIndex = 36;
			this->WAVFileName_label->Text = L"WAV file name";
			// 
			// WAV_file_name_textBox
			// 
			this->WAV_file_name_textBox->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->WAV_file_name_textBox->Location = System::Drawing::Point(12, 362);
			this->WAV_file_name_textBox->Name = L"WAV_file_name_textBox";
			this->WAV_file_name_textBox->Size = System::Drawing::Size(1034, 30);
			this->WAV_file_name_textBox->TabIndex = 35;
			// 
			// TxProp_label
			// 
			this->TxProp_label->AutoSize = true;
			this->TxProp_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->TxProp_label->Location = System::Drawing::Point(394, 256);
			this->TxProp_label->Name = L"TxProp_label";
			this->TxProp_label->Size = System::Drawing::Size(232, 24);
			this->TxProp_label->TabIndex = 37;
			this->TxProp_label->Text = L"Data bit rate per packet";
			// 
			// Data_label
			// 
			this->Data_label->AutoSize = true;
			this->Data_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Data_label->Location = System::Drawing::Point(786, 9);
			this->Data_label->Name = L"Data_label";
			this->Data_label->Size = System::Drawing::Size(53, 24);
			this->Data_label->TabIndex = 39;
			this->Data_label->Text = L"Data";
			// 
			// Data_comboBox
			// 
			this->Data_comboBox->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->Data_comboBox->FormattingEnabled = true;
			this->Data_comboBox->Location = System::Drawing::Point(790, 44);
			this->Data_comboBox->Name = L"Data_comboBox";
			this->Data_comboBox->Size = System::Drawing::Size(63, 31);
			this->Data_comboBox->TabIndex = 38;
			this->Data_comboBox->SelectedIndexChanged += gcnew System::EventHandler(this, &MyForm::Data_comboBox_SelectedIndexChanged);
			// 
			// RawTx_label
			// 
			this->RawTx_label->AutoSize = true;
			this->RawTx_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->RawTx_label->Location = System::Drawing::Point(394, 173);
			this->RawTx_label->Name = L"RawTx_label";
			this->RawTx_label->Size = System::Drawing::Size(123, 24);
			this->RawTx_label->TabIndex = 44;
			this->RawTx_label->Text = L"Raw bit rate";
			// 
			// rawkbpslabel
			// 
			this->rawkbpslabel->AutoSize = true;
			this->rawkbpslabel->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->rawkbpslabel->Location = System::Drawing::Point(498, 211);
			this->rawkbpslabel->Name = L"rawkbpslabel";
			this->rawkbpslabel->Size = System::Drawing::Size(52, 23);
			this->rawkbpslabel->TabIndex = 41;
			this->rawkbpslabel->Text = L"kbps";
			// 
			// raw_kbps_textBox
			// 
			this->raw_kbps_textBox->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->raw_kbps_textBox->Location = System::Drawing::Point(398, 208);
			this->raw_kbps_textBox->Name = L"raw_kbps_textBox";
			this->raw_kbps_textBox->Size = System::Drawing::Size(90, 30);
			this->raw_kbps_textBox->TabIndex = 40;
			// 
			// data_transmitted_label
			// 
			this->data_transmitted_label->AutoSize = true;
			this->data_transmitted_label->Font = (gcnew System::Drawing::Font(L"Arial", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(177)));
			this->data_transmitted_label->Location = System::Drawing::Point(650, 256);
			this->data_transmitted_label->Name = L"data_transmitted_label";
			this->data_transmitted_label->Size = System::Drawing::Size(167, 24);
			this->data_transmitted_label->TabIndex = 45;
			this->data_transmitted_label->Text = L"Data transmitted";
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1173, 425);
			this->Controls->Add(this->data_transmitted_label);
			this->Controls->Add(this->RawTx_label);
			this->Controls->Add(this->rawkbpslabel);
			this->Controls->Add(this->raw_kbps_textBox);
			this->Controls->Add(this->Data_label);
			this->Controls->Add(this->Data_comboBox);
			this->Controls->Add(this->TxProp_label);
			this->Controls->Add(this->WAVFileName_label);
			this->Controls->Add(this->WAV_file_name_textBox);
			this->Controls->Add(this->Tx_wav_checkBox);
			this->Controls->Add(this->Tx_Precentage_label);
			this->Controls->Add(this->Tx_Progress_textBox);
			this->Controls->Add(this->kbpslabel);
			this->Controls->Add(this->kbps_textBox);
			this->Controls->Add(this->Tx_test_stream_checkBox);
			this->Controls->Add(this->MFMhz_label);
			this->Controls->Add(this->MeasuredTx_freq_label);
			this->Controls->Add(this->Measured_freq_textBox);
			this->Controls->Add(this->Shift_checkBox);
			this->Controls->Add(this->CFMHz_label);
			this->Controls->Add(this->Center_freq_label);
			this->Controls->Add(this->Center_Freq_textBox);
			this->Controls->Add(this->BCMHz_label);
			this->Controls->Add(this->Base_clock_label);
			this->Controls->Add(this->BaseShiftnumeric);
			this->Controls->Add(this->MBCMhz_label);
			this->Controls->Add(this->Memory_base_label);
			this->Controls->Add(this->Memory_base_textBox);
			this->Controls->Add(this->Divider_label);
			this->Controls->Add(this->Divider_comboBox);
			this->Controls->Add(this->MCMhz_label);
			this->Controls->Add(this->Memory_clock_label);
			this->Controls->Add(this->Memory_clock_textBox);
			this->Controls->Add(this->GPUs_list_label);
			this->Controls->Add(this->GPUsList);
			this->Name = L"MyForm";
			this->Text = L"Scotty";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->BaseShiftnumeric))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

	// Declare variables

	// Declare a variable containing the number of the active GPU in the GPU list
	int SelectedGPU = 0;

	// Declare a variable containing the RAM frequency value for the selected GPU
	float RAMFreq = 0;

	// Declare a variable containing the calculated base RAM frequency for the selected GPU
	float BaseRAMFreq = 0;

	// Declare a variable containing the calculated shifted RAM frequency for the selected GPU
	float CenterFreq = 0;

	// Declare a variable containing the RAM memory divider value for the selected GPU
	int MemoryClockDivider = 1;

	// Declare a variable than contains the required action to be performed by the TxThread function
	int TxThreadAction = 0; // 0 = Do nothing. 1 = Transmit test pattern. 2 = Transmit audio.

	// Declare a variable than contains the transmission progress
	int TxThreadProgress = 0; // [%]

	// Declare a variable than contains the raw bit rate of transmitted data (including all bytes)
	int RawTxRate = 0; // [bps]

	// Declare a variable than contains the payload bit rate
	int TxRate = 0; // [bps]

	private: System::Windows::Forms::Timer^  InnerTimer;

	private:

		// Browse and select a WAV file name function
		void SelectFile(void)
		{
			OpenFileDialog^ openFileDialog1 = gcnew OpenFileDialog;
			//openFileDialog1->InitialDirectory = "e:\\";
			openFileDialog1->Title = "open files";
			openFileDialog1->Filter = "WAV files (*.wav)|*.wav";
			openFileDialog1->FilterIndex = 1;
			openFileDialog1->Multiselect = false;
			char Filename[1024] = {};
			char pathFilename[1024] = {};
			//
			//

			if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK)
			{
				WAV_file_name_textBox->Text = openFileDialog1->FileName;
			}
		}

		// Place default WAV file name in the file name box
		void InitWAVFileName(void)
		{
			WAV_file_name_textBox->Text = "C:\\John F. Kennedy at Rice University.wav";
		}

		// Initialize the values of the "GPUs list" list and select the default value
		void InitTable(void)
		{
			// Fill the cards list
			for (unsigned int i = 0; i < NumberOfPGUs; ++i)
			{
				char FullName[64];
				
				char *ps = GPUsFullName + 64 * i;
				strncpy_s(FullName, 64, ps, 64);

				GPUsList->Items->Add(gcnew String(FullName));
			}
			GPUsList->SelectedIndex = SelectedGPU;
		}

		// Initialize the values of the "Divider" drop list and select the default value
		void InitDivider(void)
		{
			// Fill the divider list
			for (unsigned int i = 0; i < NumberOfPGUs; ++i)
			{
				Divider_comboBox->Items->Add("1");
				Divider_comboBox->Items->Add("2");
				Divider_comboBox->Items->Add("4");
			}
			Divider_comboBox->SelectedIndex = 0;
		}

		// Initialize the values of the "Data" drop list and select the default value
		void InitData(void)
		{
			// Fill the data list
			Data_comboBox->Items->Add(MemorySet0.ToString("X2"));
			Data_comboBox->Items->Add(MemorySet1.ToString("X2"));
			Data_comboBox->Items->Add(MemorySet2.ToString("X2"));
			Data_comboBox->SelectedIndex = 0;
		}

		// Transmit a byte, 2 symbols per byte (4 bits per symbol / 2 nibbles per byte)
		void TxByte2SymbolsPerByte(byte TxDataByte)
		{
			// 2 symbols per byte / 4 bits per symbol
			// Transmit most significant nibble
			int TxLength = ((int)((TxDataByte >> 4) + ZeroOffset)) * MemoryBlockSize;
			cudaMemcpy(device_array_dest, device_array_source, TxLength, cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			// Transmit least significant nibble
			TxLength = ((int)((TxDataByte & 0x0F) + ZeroOffset)) * MemoryBlockSize;
			cudaMemcpy(device_array_dest, device_array_source, TxLength, cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
		}

		// Transmit a byte, 4 symbols per byte (2 bits per symbol)
		void TxByte4SymbolsPerByte(byte TxDataByte)
		{
			// 4 symbols per byte / 2 bits per symbol
			// Transmit most significant 2 bits
			int TxLength = ((int)((TxDataByte >> 6) + ZeroOffset)) * MemoryBlockSize;
			cudaMemcpy(device_array_dest, device_array_source, TxLength, cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			// Transmit 2nd significant 2 bits
			TxLength = ((int)(((TxDataByte >> 4) & 0x03) + ZeroOffset)) * MemoryBlockSize;
			cudaMemcpy(device_array_dest, device_array_source, TxLength, cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			// Transmit 3rd significant 2 bits
			TxLength = ((int)(((TxDataByte >> 2) & 0x03) + ZeroOffset)) * MemoryBlockSize;
			cudaMemcpy(device_array_dest, device_array_source, TxLength, cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			// Transmit least significant 2 bits
			TxLength = ((int)((TxDataByte & 0x03) + ZeroOffset)) * MemoryBlockSize;
			cudaMemcpy(device_array_dest, device_array_source, TxLength, cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
		}

		// Transmit packet without using FEC
		void TxPacket(byte *DataBuffer, int PresentPointer)
		{
			// Transmit the data, according to the number of symbols per byte (4 or 2)
			if (SymbolsPerByte == 4)
			{
				// Transmit header
				TxByte4SymbolsPerByte(TxHeader1);
				TxByte4SymbolsPerByte(TxHeader2);
				TxByte4SymbolsPerByte(TxHeader3);
				TxByte4SymbolsPerByte(TxHeader4);

				// Transmit data and calculate checksum
				int unsigned TxChecksum = 0;
				for (int i = 0; i < PacketLength; i++)
				{
					// Transmit data
					byte DataByte = DataBuffer[PresentPointer + i];
					TxByte4SymbolsPerByte(DataByte);

					// Calculate checksum
					TxChecksum += DataByte;
				}

				// Transmit checksum
				TxByte4SymbolsPerByte((TxChecksum >> 8) & 0xFF);
				TxByte4SymbolsPerByte(TxChecksum & 0xFF);
			}
			else
			{
				// Transmit header
				TxByte2SymbolsPerByte(TxHeader1);
				TxByte2SymbolsPerByte(TxHeader2);
				TxByte2SymbolsPerByte(TxHeader3);
				TxByte2SymbolsPerByte(TxHeader4);

				// Transmit data and calculate checksum
				int unsigned TxChecksum = 0;
				for (int i = 0; i < PacketLength; i++)
				{
					// Transmit data
					byte DataByte = DataBuffer[PresentPointer + i];
					TxByte2SymbolsPerByte(DataByte);

					// Calculate checksum
					TxChecksum += DataByte;
				}

				// Transmit checksum
				TxByte2SymbolsPerByte((TxChecksum >> 8) & 0xFF);
				TxByte2SymbolsPerByte(TxChecksum & 0xFF);
			}
		}

		// Transmit packet with FEC
		void TxPacketWithFEC(byte *DataBuffer, int PresentPointer)
		{
			// Prepare the data and checksum for the reed solomon FEC encoder
			int unsigned TxChecksum = 0;
			for (int i = 0; i < PacketLength; i++)
			{
				int PacketItem = (int)DataBuffer[PresentPointer + i];
				FECdata[i] = PacketItem;
				TxChecksum += PacketItem;
			}
			FECdata[PacketLength] = (TxChecksum >> 8) & 0xFF;
			FECdata[PacketLength + 1] = TxChecksum & 0xFF;

			// Encode data[] to produce parity in bb[].  Data input and parity output is in polynomial form
			encode_rs();

			// Transmit the data, according to the number of symbols per byte (4 or 2)
			if (SymbolsPerByte == 4)
			{
				// Transmit header
				TxByte4SymbolsPerByte(TxHeader1);
				TxByte4SymbolsPerByte(TxHeader2);
				TxByte4SymbolsPerByte(TxHeader3);
				TxByte4SymbolsPerByte(TxHeader4);

				// Transmit FEC parity bytes
				for (int i = 0; i < nn - kk; i++) TxByte4SymbolsPerByte((byte)bb[i]);

				// Transmit data
				for (int i = 0; i < PacketLength + 2; i++) TxByte4SymbolsPerByte((byte)FECdata[i]);
			}
			else
			{
				// Transmit header
				TxByte2SymbolsPerByte(TxHeader1);
				TxByte2SymbolsPerByte(TxHeader2);
				TxByte2SymbolsPerByte(TxHeader3);
				TxByte2SymbolsPerByte(TxHeader4);

				// Transmit FEC parity bytes
				for (int i = 0; i < nn - kk; i++)  TxByte2SymbolsPerByte((byte)bb[i]);

				// Transmit data
				for (int i = 0; i < PacketLength + 2; i++) TxByte2SymbolsPerByte((byte)FECdata[i]);
			}
		}

		// Transmit data (raw bytes - not audio)
		void TxData(byte *DataBuffer, unsigned long int DataBufferLength, bool useFEC)
		{
			unsigned long int FullPacketsBytesCount = (DataBufferLength / PacketLength) * PacketLength;
			unsigned long int p = 0;

			// Transmit packets
			for (p = 0; p < FullPacketsBytesCount; p = p + PacketLength)
			{
				// Check if the checkbox had been cleared. If so - break;
				if (Tx_test_stream_checkBox->Checked == false) break;

				// Check if EndTx flag was raised. If so - break
				if (EndTx) break;

				// Get present time
				high_resolution_clock::time_point t0 = high_resolution_clock::now();

				// Transmit packet
				if (useFEC) TxPacketWithFEC(DataBuffer, p);
				else TxPacket(DataBuffer, p);

				// Calculate time diff.
				high_resolution_clock::time_point t1 = high_resolution_clock::now();
				duration<double, std::micro> time_span = t1 - t0;
				int Time_interval = Convert::ToInt32(time_span.count());

				// Updated Tx rate
				if (p == 0)
				{
					RawTxRate = (4 + PacketLength + 2 + nn - kk) * 8 * 1000000 / Time_interval;
					TxRate = (PacketLength - 1) * 8 * 1000000 / Time_interval;
				}
				else
				{
					RawTxRate = (RawTxRate * 15 + (4 + PacketLength + 2 + nn - kk) * 8 * 1000000 / Time_interval) / 16;
					TxRate = (TxRate * 15 + (PacketLength - 1) * 8 * 1000000 / Time_interval) / 16;
				}

				// Updated progress
				TxThreadProgress = 100 * p / DataBufferLength;
			}

			// Transmit last packet if needed
			if ((p < (unsigned long int)DataBufferLength) & Tx_test_stream_checkBox->Checked & (EndTx == false))
			{
				//Declare last packet ata buffer
				byte *LastDataBuffer;
				LastDataBuffer = (byte*)malloc(PacketLength);

				// Prepare last data buffer.
				int i = 0;
				for (p = p; p < (unsigned long int)DataBufferLength; p++)
				{
					LastDataBuffer[i] = DataBuffer[p];
					i++;
				}

				// Fill the rest of the bytes with zeros.
				for (i = i; i < PacketLength; i++)
				{
					LastDataBuffer[i] = 0x00;
				}

				// Transmit last packet
				TxPacket(LastDataBuffer, 0);

				// Free last packet data buffer
				free(LastDataBuffer);
			}

			// Report thread progress complited
			TxThreadProgress = 100;
		}

		// Transmit audio
		void TxVoice(short int *PCMDataBuffer, unsigned long int DataBufferLength, float PCMSampleRate, int ChannelsNum, bool useFEC)
		{
			byte EncodedPacket[PacketLength];
			int EncodedPacketIndex = 0;
			int TempCompressionReg = 0;
			int ShiftIndex = 0;
			unsigned char PacketCounter = 0;
			unsigned int TimeThreshold = 0;

			// Verify that the WAV file sample rate is equal to or higher than the selected sampling rate
			if (PCMSampleRate >= CodecSamplingRate)
			{
				// Initialize the G.726 state
				g726_state G726StatePointer;
				g726_init_state(&G726StatePointer);

				float SampleRatio = PCMSampleRate / CodecSamplingRate;
				float SampleRatioIndex = 1;

				float LPFAlpha = LowPassFilterFrequency / PCMSampleRate;
				
				float FilteredData = PCMDataBuffer[0];
				float PrevFilteredData;
				float UnderSampledData;

				int EncodedData = 0;

				unsigned long DataIndex = 0;

				// Get present time
				high_resolution_clock::time_point t0 = high_resolution_clock::now();
				high_resolution_clock::time_point tRef = t0;

				// Prepare delta to 1 second threshold parameter, in milliseconds
				high_resolution_clock::time_point tThreshold = tRef + std::chrono::milliseconds(1000);

				// Downsampling loop
				while (DataIndex < DataBufferLength)
				{
					// Check if the checkbox had been cleared. If so - break;
					if (Tx_wav_checkBox->Checked == false) break;

					// Check if EndTx flag was raised. If so - break
					if (EndTx) break;

					// Check if a new packet is set. If so - add protocol bytes, as needed
					if (EncodedPacketIndex == 0)
					{
						// Set the first byte to be PacketCounter
						EncodedPacket[EncodedPacketIndex] = (byte)PacketCounter;
						EncodedPacketIndex++;
					}
					
					// Apply a low pass filter on the data
					PrevFilteredData = FilteredData;
					FilteredData += LPFAlpha * ((float)PCMDataBuffer[DataIndex] - FilteredData);

					// Reduce down sampling index by one
					SampleRatioIndex -= 1;

					// Check if a down sampled data is to be taken.
					// If so - perform a sample using a linear interpolation
					if (SampleRatioIndex < 0)
					{
						// Take the undersampled data
						UnderSampledData = SampleRatioIndex * (FilteredData - PrevFilteredData) + FilteredData;

						// Advance down sampling index by SampleRatio
						SampleRatioIndex += SampleRatio;

						// Encode the under sampled data
						switch (CodecCompression)
						{
						case 3:
							EncodedData = g726_24_encoder((int)UnderSampledData, AUDIO_ENCODING_LINEAR, &G726StatePointer);
							break;
						case 4:
							EncodedData = g726_32_encoder((int)UnderSampledData, AUDIO_ENCODING_LINEAR, &G726StatePointer);
							break;
						case 5:
							EncodedData = g726_40_encoder((int)UnderSampledData, AUDIO_ENCODING_LINEAR, &G726StatePointer);
							break;
						default: // Assuming 2 bits per channel was selected (16,000 bps)
							EncodedData = g726_16_encoder((int)UnderSampledData, AUDIO_ENCODING_LINEAR, &G726StatePointer);
						}

						//Compress the data into bytes
						TempCompressionReg = (TempCompressionReg << CodecCompression) | EncodedData;
						ShiftIndex += CodecCompression;

						// If 8 bits had been reached then store the data and zero the temporary registers
						if (ShiftIndex == 8)
						{
							EncodedPacket[EncodedPacketIndex] = (byte) TempCompressionReg;
							TempCompressionReg = 0;
							ShiftIndex = 0;
							EncodedPacketIndex++;
						}
						// If 8 bits had been reached then store the data and prepare the temporary registers for further work
						else if (ShiftIndex > 8)
						{
							ShiftIndex -= 8;
							EncodedPacket[EncodedPacketIndex] = (byte) (TempCompressionReg >> ShiftIndex);
							TempCompressionReg = TempCompressionReg & ( (1 << ShiftIndex) - 1);
							EncodedPacketIndex++;
						}
					}

					//Advance DataIndex
					DataIndex += ChannelsNum;

					// Check is this is the end of the last packet in the packet cycle (Every 1 second).
					// If so - add reserved protocol bytes, as needed
					if (((PacketCounter % PacketsPerSecond) == (PacketsPerSecond - 1)) & 
						(EncodedPacketIndex == (PacketLength - ReservedBytesPerSecond)))
							while (EncodedPacketIndex < PacketLength) EncodedPacket[EncodedPacketIndex++] = 0;
					// Check is the last data had been read. If so - fill the rest with 0
					else if (DataIndex >= DataBufferLength)
						while (EncodedPacketIndex < PacketLength) EncodedPacket[EncodedPacketIndex++] = 0;

					// Check if there's a complete packet to transmit
					if (EncodedPacketIndex == PacketLength)
					{
						// Transmit packet
						if (useFEC) TxPacketWithFEC(EncodedPacket, 0);
						else TxPacket(EncodedPacket, 0);

						// Calculate time diff. for data rate calculations
						high_resolution_clock::time_point t1 = high_resolution_clock::now();
						duration<double, std::micro> time_span = t1 - t0;
						int Time_interval = Convert::ToInt32(time_span.count());

						// Get present time
						t0 = t1;

						// Updated Tx rate
						if (DataIndex == ChannelsNum)
						{
							RawTxRate = (4 + PacketLength + 2 + nn - kk) * 8 * 1000000 / Time_interval;
							TxRate = (PacketLength - 1) * 8 * 1000000 / Time_interval;
						}
						else
						{
							RawTxRate = (RawTxRate * 7 + (4 + PacketLength + 2 + nn - kk) * 8 * 1000000 / Time_interval) / 8;
							TxRate = (TxRate * 7 + (PacketLength - 1) * 8 * 1000000 / Time_interval) / 8;
						}

						// Updated progress
						TxThreadProgress = 100 * DataIndex / DataBufferLength;

						// Zero EncodedPacketIndex
						EncodedPacketIndex = 0;

						// Advance the packet counter
						if (PacketCounter == PacketPerUnsignedCharThreshold) PacketCounter = 0;
						else PacketCounter++;

						// Check if last packet of the current second had been transmitted.
						// If so - wait for the second to elapse (according to adaptive threshold)
						if ((PacketCounter % PacketsPerSecond) == 0)
						{
							// Get a fresh time tage
							t1 = high_resolution_clock::now();

							// If the total time had passed the threshold, calculate the new 
							// threshold and begin a new packets cycle in a new second
							while (t1 < tThreshold)
							{
								t1 = high_resolution_clock::now();
							}

							// Calculate new time threshold
							duration<double, std::milli> time_span = t1 - tRef;
							TimeThreshold += 1000 - Convert::ToInt32(time_span.count());
							tRef = t1;
							tThreshold = tRef + std::chrono::milliseconds(1000 + TimeThreshold);
						}
					}
				}
			}

			// Report thread progress complited
			TxThreadProgress = 100;
		}

		// Transmission thread
		void TxThread()
		{
			InTx = true;

			// Check if the thread is required to tranmit data
			if (TxThreadAction == 1)
			{
				byte *DataBuffer;
				const int DataBufferLength = PacketLength * 1000 + 1;
				int FullPacketsBytesCount = (DataBufferLength / PacketLength) * PacketLength;

				// Generate test data
				DataBuffer = (byte*) malloc(DataBufferLength);
				
				int q = 0;
				while (q < FullPacketsBytesCount)
				{
					int TestData = 0x30;
					for (int i = 0; i < 16; i++)
					{
						DataBuffer[q] = TestData + i;
						q++;
					}
				}
				while (q < DataBufferLength)
				{
					DataBuffer[q] = 0x00;
					q++;
				}

				// Transmit the data
				TxData(DataBuffer, DataBufferLength, useFEC);

				// Free memory
				free(DataBuffer);
			}
			else
				// Check if the thread is required to tranmit audio
				if (TxThreadAction == 2)
				{
					// Open the WAV file
					FILE *fin;
					IntPtr ptrToFileName = Marshal::StringToHGlobalAnsi(WAV_file_name_textBox->Text);
					errno_t err = fopen_s(&fin, static_cast<char*>(ptrToFileName.ToPointer()), "rb");
					Marshal::FreeHGlobal(ptrToFileName);

					// If the WAV file is opened then continue reading the file
					if (fin != NULL) {
						//Read WAV header
						wav_header_t header;
						fread(&header, sizeof(header), 1, fin);

						//Reading file
						chunk_t chunk;

						//Go to the data chunk
						while (true)
						{
							fread(&chunk, sizeof(chunk), 1, fin);
							// Check if the data part was found reached
							if (*(unsigned int *)&chunk.ID == 0x61746164)
								break;
							//skip chunk data bytes
							fseek(fin, chunk.size, SEEK_CUR);
						}

						//Number of samples
						int sample_size = header.bitsPerSample / 8;
						int samples_count = chunk.size * 8 / header.bitsPerSample;

						short int *WAVPCMData = new short int[samples_count];
						memset(WAVPCMData, 0, sizeof(short int) * samples_count);

						//Reading data
						int ReadResult = (int)fread(WAVPCMData, sample_size * samples_count, 1, fin);

						// Close file
						fclose(fin);

						// Transmit the PCM data
						TxVoice(WAVPCMData, samples_count, (float)header.sampleRate, header.numChannels, useFEC);
					}
				}

			// Mark that transmittion had ended
			InTx = false;
			EndTx = false;
		}

		// Initialize the transmission thread
		void InitTxThread(void)
		{
			// Set progress to 0;
			TxThreadProgress = 0;

			System::Threading::Thread^ t1;
			System::Threading::ThreadStart^ ts = gcnew System::Threading::ThreadStart(this, &MyForm::TxThread);
			t1 = gcnew System::Threading::Thread(ts);
			t1->Start();
		}

		// Update form fields
		void UpdateFields(void)
		{
			int Divider_comboBox_index = Divider_comboBox->SelectedIndex;
			switch (Divider_comboBox_index)
			{
			case 0: // 0 = 1
				MemoryClockDivider = 1;
				break;
			case 1: // 1 = 2
				MemoryClockDivider = 2;
				break;
			case 2: // 2 = 4
				MemoryClockDivider = 4;
				break;
			default: // 0 = 1
				MemoryClockDivider = 1;
			}
			BaseRAMFreq = RAMFreq / (float)MemoryClockDivider;
			Memory_base_textBox->Text = Convert::ToString(BaseRAMFreq);
			float FreqShift = Convert::ToSingle(BaseShiftnumeric->Value);
			CenterFreq = BaseRAMFreq + FreqShift;
			Center_Freq_textBox->Text = Convert::ToString(CenterFreq);
		}

		// Initiate an internal tick loop, to perform cyclic actions within the form
		void InitInnerTimer(int TickIntervals)
		{
			this->components = (gcnew System::ComponentModel::Container());
			this->InnerTimer = (gcnew System::Windows::Forms::Timer(this->components));
			this->InnerTimer->Enabled = true;
			this->InnerTimer->Interval = TickIntervals;
			this->InnerTimer->Tick += gcnew System::EventHandler(this, &MyForm::InnerTimer_Tick);
		}

		// Set the memory frequency using the NvApi
		int SetFrequency(int FreqShiftKhz)
		{
			// Set attributes
			NV_GPU_PERF_PSTATES20_INFO pStatesInfo = { 0 };

			pStatesInfo.version = MAKE_NVAPI_VERSION(pStatesInfo, 2);
			pStatesInfo.numPStates = 1;
			pStatesInfo.numClocks = 1;
			pStatesInfo.pStates[0].pStateId = 0;
			pStatesInfo.pStates[0].clocks[0].domainId = 4;
			pStatesInfo.pStates[0].clocks[0].typeId = 0;
			pStatesInfo.pStates[0].clocks[0].frequencyDeltaKHz.value = FreqShiftKhz;

			return(NvAPI_GPU_SetPstates20(NvApiGpuHandles[BusId[SelectedGPU]], &pStatesInfo));
		}

		// Internal form tick loop
	private: System::Void InnerTimer_Tick(System::Object^  sender, System::EventArgs^  e) {
		// Get measured memory clock frequency
		NV_GPU_CLOCK_FREQUENCIES clkFreqs = { 0 };
		clkFreqs.version = NV_GPU_CLOCK_FREQUENCIES_VER;
		clkFreqs.ClockType = NV_GPU_CLOCK_FREQUENCIES_CURRENT_FREQ;
		int result = NvAPI_GPU_GetAllClockFrequencies(NvApiGpuHandles[BusId[SelectedGPU]], &clkFreqs);
		if (result == 0)
		{
			float MeasuredFreq = (float)(clkFreqs.domain[NVAPI_GPU_PUBLIC_CLOCK_MEMORY].frequency) / (float) 1000.0 / (float)MemoryClockDivider;
			Measured_freq_textBox->Text = Convert::ToString(MeasuredFreq);
		}
		else
		{
			Measured_freq_textBox->Text = "0";
		}

		// Update Tx progress
		Tx_Progress_textBox->Text = Convert::ToString(TxThreadProgress);

		// Update raw data Tx rate
		raw_kbps_textBox->Text = Convert::ToString((double) RawTxRate / 1000.0);

		// Update data Tx rate
		kbps_textBox->Text = Convert::ToString((double) TxRate / 1000.0);

		// If transmission progress passed 99, then it's set on 100, meaning that the tramsission is over.
		if (TxThreadProgress > 99)
		{
			Tx_test_stream_checkBox->Checked = false;
			Tx_wav_checkBox->Checked = false;
		}
	}

			 // GPUsList value changed
	private: System::Void GPUsList_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		// Read selected GPU
		SelectedGPU = GPUsList->SelectedIndex;

		// Read attributes
//		NV_GPU_PERF_PSTATES20_INFO pStatesInfo = { 0 };
//		pStatesInfo.version = MAKE_NVAPI_VERSION(pStatesInfo, 2);
//		int result = NvAPI_GPU_GetPstates20(NvApiGpuHandles[BusId[SelectedGPU]], &pStatesInfo);
//		if (result == 0)
//		{
//			RAMFreq = (float)(pStatesInfo.pStates[0].clocks[1]).data.single.frequencyKHz / (float) 1000.0;
//			Memory_clock_textBox->Text = Convert::ToString(RAMFreq);
//		}

//		NV_GPU_CLOCK_FREQUENCIES clkFreqs = { NV_GPU_CLOCK_FREQUENCIES_VER };

//		NV_GPU_CLOCK_FREQUENCIES clkFreqs = { 0 };
//		clkFreqs.version = NV_GPU_CLOCK_FREQUENCIES_VER;
//		clkFreqs.ClockType = NV_GPU_CLOCK_FREQUENCIES_BASE_CLOCK; // NV_GPU_CLOCK_FREQUENCIES_CURRENT_FREQ;
//		int result = NvAPI_GPU_GetAllClockFrequencies(NvApiGpuHandles[BusId[SelectedGPU]], &clkFreqs);
//		if (result == 0)
//		{
//			double GpuCurrentCoreClock = clkFreqs.domain[NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS].frequency / 1000;
//			double GpuCurrentMemoryClock = clkFreqs.domain[NVAPI_GPU_PUBLIC_CLOCK_MEMORY].frequency / 1000;
//			double GpuCurrentShaderClock = clkFreqs.domain[NVAPI_GPU_PUBLIC_CLOCK_PROCESSOR].frequency / 1000;
//			Memory_clock_textBox->Text = Convert::ToString(GpuCurrentMemoryClock);
//		}

//		NV_CLOCKS_INFO clocksInfo = { NV_CLOCKS_INFO_VER };
//		int result = NvAPI_GPU_GetAllClocks(NvApiGpuHandles[BusId[SelectedGPU]], &clocksInfo);
//		if (result == 0)
//		{
//			unsigned int GpuCurrentCoreClock = clocksInfo.clocks[0] / 1000;
//			unsigned int GpuCurrentMemoryClock = clocksInfo.clocks[1] / 1000;
//			Memory_clock_textBox->Text = Convert::ToString(GpuCurrentMemoryClock);
//		}

		// Read memory clock frequency

		NV_GPU_CLOCK_FREQUENCIES clkFreqs = { 0 };
		clkFreqs.version = NV_GPU_CLOCK_FREQUENCIES_VER;
		clkFreqs.ClockType = NV_GPU_CLOCK_FREQUENCIES_BASE_CLOCK;
		int result = NvAPI_GPU_GetAllClockFrequencies(NvApiGpuHandles[BusId[SelectedGPU]], &clkFreqs);
		if (result == 0)
		{
			RAMFreq = (float)(clkFreqs.domain[NVAPI_GPU_PUBLIC_CLOCK_MEMORY].frequency) / (float) 1000.0;
			Memory_clock_textBox->Text = Convert::ToString(RAMFreq);
		}

		// Read memeory RAM type
		unsigned int RamCode = 0;
		result = NvAPI_GPU_GetRamType(NvApiGpuHandles[BusId[SelectedGPU]], &RamCode);
		if (result == 0)
		{
			switch (RamCode)
			{
			case 0: // Unknown
				MemoryClockDivider = 1;
				break;
			case 1: // SDRAM
				MemoryClockDivider = 1;
				break;
			case 2: // DDR1
				MemoryClockDivider = 2;
				break;
			case 3: // DDR2
				MemoryClockDivider = 2;
				break;
			case 4: // GDDR2
				MemoryClockDivider = 2;
				break;
			case 5: // GDDR3
				MemoryClockDivider = 2;
				break;
			case 6: // GDDR4
				MemoryClockDivider = 2;
				break;
			case 7: // DDR3
				MemoryClockDivider = 2;
				break;
			case 8: // GDDR5
				MemoryClockDivider = 2;
				break;
			case 9: // LPDDR2
				MemoryClockDivider = 2;
				break;
			case 10: // GDDR5X
				MemoryClockDivider = 4;
				break;
			case 14: // GDDR6
				MemoryClockDivider = 4;
				break;
			default:
				MemoryClockDivider = 1;
			}

			// Update memory clock divider value, accoring to the RAM type
			switch (MemoryClockDivider)
			{
			case 1:
				Divider_comboBox->SelectedIndex = 0; // 0 = 1
				break;
			case 2:
				Divider_comboBox->SelectedIndex = 1; // 1 = 2
				break;
			case 4:
				Divider_comboBox->SelectedIndex = 2; // 2 = 4
				break;
			default:
				Divider_comboBox->SelectedIndex = 0; // 0 = 1
			}

			// Update form fields
			UpdateFields();
		}
	}

	// Divider value changes
private: System::Void Divider_comboBox_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
	// Update form fields
	UpdateFields();
}

	// Shifting contril value changed
private: System::Void BaseShiftnumeric_ValueChanged(System::Object^  sender, System::EventArgs^  e) {
	// Set attributes
	int FreqShiftKhz = 0;

	// If the shift checkbox is checked, the update the FreqShiftKhz value
	if (Shift_checkBox->Checked)
	{
		FreqShiftKhz = (int)((double)1000 * Convert::ToDouble(BaseShiftnumeric->Value)) * MemoryClockDivider;
	}

	// Set the required frequency
	int result = SetFrequency(FreqShiftKhz);

	// Update form fields
	UpdateFields();
}

	// Shift checkbox value changed
private: System::Void Shift_checkBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
	// Set attributes
	int FreqShiftKhz = 0;

	// If the shift checkbox is checked, the update the FreqShiftKhz value
	if (Shift_checkBox->Checked)
	{
		FreqShiftKhz = (int)((double)1000 * Convert::ToDouble(BaseShiftnumeric->Value)) * MemoryClockDivider;
	}

	// Set the required frequency
	int result = SetFrequency(FreqShiftKhz);
}

	// TX checkbox value changed
private: System::Void Tx_checkBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
	if ((Tx_test_stream_checkBox->Checked) & (Tx_wav_checkBox->Checked == false))
	{
		// Set TxThreadAction to output a test string
		TxThreadAction = 1;

		// Start transmission
		InitTxThread();
	}
	else
	{
		// Clear TX checkbox
		Tx_test_stream_checkBox->Checked = false;
	}
}

	// TX WAV checkbox value changed
private: System::Void Tx_wav_checkBox_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
	if ((Tx_test_stream_checkBox->Checked == false) & (Tx_wav_checkBox->Checked))
	{
		// Set TxThreadAction to output a test string
		TxThreadAction = 2;

		// The the file name textbox is not set, then ask the user to select a WAV file
		if (WAV_file_name_textBox->Text == "")
		{
			SelectFile();
		}
		
		// Start transmission
		InitTxThread();
	}
	else
	{
		// Clear TX WAV checkbox
		Tx_wav_checkBox->Checked = false;
	}
}

	// Data selection value was changes
private: System::Void Data_comboBox_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
	// Set source memory according to data
	int Data_comboBox_index = Data_comboBox->SelectedIndex;
	cudaDeviceSynchronize();
	byte FillingData;

	// Set memory block value accoridng to the Data value
	switch (Data_comboBox_index)
	{
	case 1: // 1 = MemorySet1
		FillingData = MemorySet1;
		break;
	case 2: // 2 = MemorySet2
		FillingData = MemorySet2;
		break;
	default: // 0 = MemorySet0
		FillingData = MemorySet0;
	}
	cudaMemset(device_array_source, FillingData, MemoryBitsAllocation*MemoryBlockSize);
	cudaDeviceSynchronize();
}
};
}
