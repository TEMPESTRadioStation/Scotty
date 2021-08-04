# Scotty

Scotty is a demonstration software. It demonstrates how audio may be transmitted from a host computer by controlling 
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
