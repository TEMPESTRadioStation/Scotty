// Header file for Simon Rockliff' Reed-Solomon encoding-decoding code.

#define mm  8           /* RS code over GF(2**8) - change to suit */
#define nn  255         /* nn=2**mm -1   length of codeword */



// Default setup
#define tt  10          /* number of errors that can be corrected */
#define kk  235         /* kk = nn-2*tt  */

// Best bit rate setup
//#define tt  2           /* number of errors that can be corrected */
//#define kk  251         /* kk = nn-2*tt  */

// Best audio quality setup
//#define tt  2           /* number of errors that can be corrected */
//#define kk  251         /* kk = nn-2*tt  */



void generate_gf();

void gen_poly();

void encode_rs();

void decode_rs();




