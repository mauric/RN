
#include <stdio.h>
#include <string.h>

#define MASK_BASE 0xF8
#define MASK_CACHEE 0x07




//----------------------------------------------------------------------------
// Minimal function called from Python
//----------------------------------------------------------------------------

int
hello(const char *msg)
{
int result=strlen(msg);
printf("native hello() received '%s' and is about to return %d\n",
       msg,result);
return result;
}

//----------------------------------------------------------------------------
// Blend images
//----------------------------------------------------------------------------

void
imgBlend(int w1,                  // first input image (and result) width
         int h1,                  // first input image (and result) height
         const unsigned char *d1, // first input image pixels (3*w1*h1 bytes)
         int w2,                  // second input image width
         int h2,                  // second input image height
         const unsigned char *d2, // second input image pixels (3*w2*h2 bytes)
         int k,                   // cursor value within range [0;100]
         unsigned char *d)        // resulting image pixels (3*w1*h1 bytes)
{


for ( int i = 0; i < h1; ++i)
{
  for (int j = 0; j < w1; ++j)
  {
    int i2,j2;
    i2=i*h2/h1;
    j2=j*w2/w1;
    d[3*(i*w1+j)] = (d1[3*(i*w1+j)]*(1-k/100.))+(d2[3*(i2*w2+j2)]*k/100.);
    d[3*(i*w1+j)+1] = (d1[3*(i*w1+j)+1]*(1-k/100.))+(d2[3*(i2*w2+j2)+1]*k/100.);
    d[3*(i*w1+j)+2] = (d1[3*(i*w1+j)+2]*(1-k/100.))+(d2[3*(i2*w2+j2)+2]*k/100.);
  }
}

//commit 1

}

//----------------------------------------------------------------------------
// Reveal an image hiden into another one
//----------------------------------------------------------------------------

void
imgReveal(int w1,                  // input image (and result) width
          int h1,                  // input image (and result) height
          const unsigned char *d1, // input image pixels (3*w1*h1 bytes)
          int k,                   // cursor value within range [0;8]
          unsigned char *d)        // resulting image pixels (3*w1*h1 bytes)
{
(void)w1; (void)h1; (void)d1; // avoid ``unused parameter'' warnings
(void)k; (void)d;

for ( int i = 0; i < h1; ++i)
{
  for (int j = 0; j < w1; ++j)
  {

    d[3*(i * w1 + j)]   = (((d1[3 * (i * w1 + j)]) << (8 - k)));
    d[3*(i * w1 + j) + 1] = ((d1[3 * (i * w1 + j) + 1]) << (8 - k ));
    d[3*(i * w1 + j) + 2] = ((d1[3 * (i * w1 + j) + 2]) << (8 - k));
  }
}


}

//----------------------------------------------------------------------------
// Hide an image into another one
//----------------------------------------------------------------------------

void
imgHide(int w1,                  // first input image (and result) width
        int h1,                  // first input image (and result) height
        const unsigned char *d1, // first input image pixels (3*w1*h1 bytes)
        int w2,                  // second input image width
        int h2,                  // second input image height
        const unsigned char *d2, // second input image pixels (3*w2*h2 bytes)
        int k,                   // cursor value within range [0;8]
        unsigned char *d)        // resulting image pixels (3*w1*h1 bytes)
{

for ( int i = 0; i < h1; ++i)
{
  for (int j = 0; j < w1; ++j)
  {
    int i2,j2;

    i2=i*h2/h1;
    j2=j*w2/w1;

    unsigned char mask_base = 0xFF;
    mask_base = mask_base << k;

    d[3*(i*w1+j)]   = (((d1[3*(i*w1+j)]) & mask_base))   +  (((d2[3*(i2*w2+j2)]) >> (8 - k)));
    d[3*(i*w1+j)+1] = ((d1[3*(i*w1+j)+1])& mask_base) + ((d2[3*(i2*w2+j2)+1]) >> (8 - k));
    d[3*(i*w1+j)+2] = ((d1[3*(i*w1+j)+2])& mask_base )+ ((d2[3*(i2*w2+j2)+2]) >> (8 - k));
  }
}

}

