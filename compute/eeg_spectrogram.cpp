#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

#include "eeg_spectrogram.hpp"

#include <time.h>
#include <sys/time.h>

using namespace arma;

// TODO(joshblum): this needs to be a concurrent structure
static edf_hdr_struct* EDF_HDR_CACHE[EDFLIB_MAXFILES];

// should use a hashmap + linked list implementation if the
// list traversal becomes a bottleneck
void print_hdr_cache()
{
  printf("[\n");
  for (int i = 0; i < EDFLIB_MAXFILES; i++)
  {
    if (EDF_HDR_CACHE[i] == NULL)
    {
      printf("-,");
    }
    else
    {
      printf("%s,\n", EDF_HDR_CACHE[i]->path);
    }
  }
  printf("]\n");
}

edf_hdr_struct* get_hdr_cache(char *filename)
{
  for (int i = 0; i < EDFLIB_MAXFILES; i++)
  {
    if (EDF_HDR_CACHE[i] != NULL)
    {
      if (!(strcmp(filename, EDF_HDR_CACHE[i]->path)))
      {
        edf_hdr_struct* hdr = EDF_HDR_CACHE[i];
        // TODO(joshblum): probably remove this once
        // windowing is fully implemented
        for (int signal = 0; signal < hdr->edfsignals; signal++)
        {
          edfrewind(hdr->handle, signal);
        }
        return hdr;
      }
    }
  }

  return NULL;
}

void set_hdr_cache(edf_hdr_struct* hdr)
{
  for (int i = 0; i < EDFLIB_MAXFILES; i++)
  {
    if (EDF_HDR_CACHE[i] == NULL)
    {
      EDF_HDR_CACHE[i] = hdr;
      break;
    }
  }
}

void pop_hdr_cache(const char* filename)
{
  for (int i = 0; i < EDFLIB_MAXFILES; i++)
  {
    if (EDF_HDR_CACHE[i] != NULL)
    {
      if (!(strcmp(filename, EDF_HDR_CACHE[i]->path)))
      {
        free(EDF_HDR_CACHE[i]);
        EDF_HDR_CACHE[i] = NULL;
      }
    }
  }
}

unsigned long long getticks()
{
  struct timeval t;
  gettimeofday(&t, 0);
  return t.tv_sec * 1000000ULL + t.tv_usec;
}

double ticks_to_seconds(unsigned long long ticks)
{
  return ticks * 1.0e-6;
}


void log_time_diff(unsigned long long ticks)
{
  double diff = ticks_to_seconds(ticks);
  printf("Time taken %.2f seconds\n", diff);
}

void print_spec_params_t(spec_params_t* spec_params)
{
  printf("spec_params: {\n");
  printf("\tfilename: %s\n", spec_params->filename);
  printf("\tduration: %.2f\n", spec_params->duration);
  printf("\thdl: %d\n", spec_params->hdl);
  printf("\tnfft: %d\n", spec_params->nfft);
  printf("\tnstep: %d\n", spec_params->nstep);
  printf("\tshift: %d\n", spec_params->shift);
  printf("\tnsamples: %d\n", spec_params->nsamples);
  printf("\tnblocks: %d\n", spec_params->nblocks);
  printf("\tnfreqs: %d\n", spec_params->nfreqs);
  printf("\tspec_len: %d\n", spec_params->spec_len);
  printf("\tfs: %d\n", spec_params->fs);
  printf("}\n");
}

int get_data_len(edf_hdr_struct* hdr)
{
  // assume all signals have a uniform sample rate
  return hdr->signalparam[0].smp_in_file;
}

int get_nfft(int shift, int pad)
{
  return fmax(get_next_pow_2(shift) + pad, shift);
}

int get_nsamples(int data_len, int fs, float duration)
{
  return fmin(data_len, fs * 60 * 60 * duration);
}

int get_nblocks(int nsamples, int nfft, int shift)
{
  return (nsamples - nfft) / shift + 1;
}

int get_nfreqs(int nfft)
{
  return nfft / 2 + 1;
}

int get_next_pow_2(unsigned int v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

int get_fs(edf_hdr_struct* hdr)
{
  return ((double)hdr->signalparam[0].smp_in_datarecord / (double)hdr->datarecord_duration) * EDFLIB_TIME_DIMENSION;
}

void get_eeg_spectrogram_params(spec_params_t* spec_params,
                                char* filename, float duration)
{
  // TODO(joshblum): implement full multitaper method
  // and remove hard coding
  spec_params->filename = filename;
  spec_params->duration = duration;

  edf_hdr_struct* hdr = (edf_hdr_struct*) malloc(sizeof(edf_hdr_struct));
  load_edf(hdr, filename);
  spec_params->hdl = hdr->handle;
  // check for errors
  if (hdr->filetype < 0)
  {
    spec_params->hdl = -1;
    spec_params->fs = 0;
    spec_params->shift = 0;
    spec_params->nstep = 0;
    spec_params->nsamples = 0;
    spec_params->nblocks = 0;
    spec_params->nfreqs = 0;
    spec_params->spec_len = 0;
    return;
  }

  spec_params->fs = get_fs(hdr);

  int data_len = get_data_len(hdr);
  int pad = 0;
  spec_params->shift = spec_params->fs * 4;
  spec_params->nstep = spec_params->fs * 1;
  spec_params->nfft = get_nfft(spec_params->shift, pad);
  spec_params->nsamples = get_nsamples(data_len, spec_params->fs, duration);
  spec_params->nblocks = get_nblocks(spec_params->nsamples,
                                     spec_params->shift, spec_params->nstep);
  spec_params->nfreqs = get_nfreqs(spec_params->nfft);
  spec_params->spec_len = spec_params->nsamples / spec_params->fs;
}

void load_edf(edf_hdr_struct* hdr, char* filename)
{
  edf_hdr_struct* cached_hdr = get_hdr_cache(filename);
  if (cached_hdr != NULL)
  {
    // get the file from the cache
    hdr = cached_hdr;
    return;
  }
  if (edfopen_file_readonly(filename, hdr, EDFLIB_READ_ALL_ANNOTATIONS))
  {
    switch (hdr->filetype)
    {
    case EDFLIB_MALLOC_ERROR                :
      printf("\nmalloc error\n\n");
      break;
    case EDFLIB_NO_SUCH_FILE_OR_DIRECTORY   :
      printf("\ncannot open file, no such file or directory: %s\n\n", filename);
      break;
    case EDFLIB_FILE_CONTAINS_FORMAT_ERRORS :
      printf("\nthe file is not EDF(+) or BDF(+) compliant\n"
             "(it contains format errors)\n\n");
      break;
    case EDFLIB_MAXFILES_REACHED            :
      printf("\nto many files opened\n\n");
      break;
    case EDFLIB_FILE_READ_ERROR             :
      printf("\na read error occurred\n\n");
      break;
    case EDFLIB_FILE_ALREADY_OPENED         :
      printf("\nfile has already been opened\n\n");
      break;
    default                                 :
      printf("\nunknown error\n\n");
      break;
    }
  }
  // set the file in the cache
  set_hdr_cache(hdr);
}

void close_edf(char* filename)
{
  edf_hdr_struct* hdr = get_hdr_cache(filename);
  if (hdr != NULL)
  {
    edfclose_file(hdr->handle);
    pop_hdr_cache(filename);
  }
}

void cleanup_spectrogram(char* filename, float* spec_arr)
{
  close_edf(filename);
  free(spec_arr);
}

float* create_buffer(int n)
{
  float* buf = (float*) malloc(sizeof(float) * n);
  if (buf == NULL)
  {
    printf("\nmalloc error\n");
  }
  return buf;
}

int read_samples(int hdl, int ch, int n, float *buf)
{
  int bytes_read = edfread_physical_samples(hdl, ch, n, buf);

  if (bytes_read == -1)
  {
    printf("\nerror: edf_read_physical_samples()\n");
    edfclose_file(hdl);
    free(buf);
    return -1;
  }
  // clear buffer in case we didn't read as much as we expected to
  for (int i = bytes_read; i < n; i++)
  {
    buf[i] = 0.0;
  }
  return bytes_read;
}

// Create a hamming window of windowLength samples in buffer
void hamming(int windowLength, float* buf)
{
  for (int i = 0; i < windowLength; i++)
  {
    buf[i] = 0.54 - (0.46 * cos( 2 * M_PI * (i / ((windowLength - 1) * 1.0))));
  }
}

static inline float abs(fftw_complex* arr, int i)
{
  return sqrt(arr[i][0] * arr[i][0] + arr[i][1] * arr[i][1]);
}

// copy pasta http://ofdsp.blogspot.co.il/2011/08/short-time-fourier-transform-with-fftw3.html
/*
 * Fill the `spec_mat` matrix with values for the spectrogram for the given diff.
 * `spec_mat` is expected to be initialized and the results are added to allow averaging
 */
void STFT(frowvec& diff, spec_params_t* spec_params, fmat& spec_mat)
{
  fftw_complex    *data, *fft_result;
  fftw_plan       plan_forward;
  int nfft = spec_params->nfft;

  int nstep = spec_params->nstep;

  int nblocks = spec_params->nblocks;
  int nfreqs = spec_params->nfreqs;
  int nsamples = spec_params->nsamples;

  data = ( fftw_complex* ) fftw_malloc( sizeof( fftw_complex ) * nfft);
  fft_result = ( fftw_complex* ) fftw_malloc( sizeof( fftw_complex ) * nfft);
  // TODO keep plans in memory until end, create plans once and cache?
  // TODO look into using arma memptr instead of copying data
  plan_forward = fftw_plan_dft_1d(nfft, data, fft_result,
                                  FFTW_FORWARD, FFTW_ESTIMATE);

  // Create a hamming window of appropriate length
  float window[nfft];
  hamming(nfft, window);

  for (int idx = 0; idx < nblocks; idx++)
  {
    // get the last chunk
    if (idx * nstep + nfft > nsamples)
    {
      int upper_bound = nsamples - idx * nstep;
      for (int i = 0; i < upper_bound; i++)
      {
        data[i][0] = diff(idx * nstep + i) * window[i];
        data[i][1] = 0.0;

      }
      for (int i = upper_bound; i < nfft; i++)
      {
        data[i][0] = 0.0;
        data[i][1] = 0.0;
      }
      break;
    }
    else
    {
      for (int i = 0; i < nfft; i++)
      {
        // TODO vector multiplication?
        data[i][0] = diff(idx * nstep  + i) * window[i];
        data[i][1] = 0.0;
      }
    }

    // Perform the FFT on our chunk
    fftw_execute( plan_forward );

    // TODO: change maybe?
    // http://www.fftw.org/fftw2_doc/fftw_2.html
    for (int i = 0; i < nfreqs; i++)
    {
      spec_mat(idx, i) += abs(fft_result, i) / nfft;
    }

    // Uncomment to see the raw-data output from the FFT calculation
    // printf("[ ");
    // for (int i = 0 ; i < 10 ; i++ )
    // {
    //   printf("%2.2f ", specs(i, idx));
    // }
    // printf("]\n");
  }

  fftw_destroy_plan(plan_forward);

  fftw_free(data);
  fftw_free(fft_result);
}

void eeg_file_spectrogram_handler(char* filename, float duration, int ch, fmat& spec_mat)
{
  spec_params_t spec_params;
  get_eeg_spectrogram_params(&spec_params, filename, duration);
  print_spec_params_t(&spec_params);
  eeg_spectrogram(&spec_params, ch, spec_mat);
}

void eeg_spectrogram_as_arr(spec_params_t* spec_params, int ch, float* spec_arr)
{
  if (spec_arr == NULL)
  {
    spec_arr = (float*) malloc(sizeof(float) * spec_params->nblocks * spec_params->nfreqs);

  }
  fmat spec_mat;
  eeg_spectrogram(spec_params, ch, spec_mat);
  serialize_spec_mat(spec_params, spec_mat, spec_arr);
}

void eeg_spectrogram(spec_params_t* spec_params, int ch, fmat& spec_mat)
{
  if (spec_params->hdl == -1)
  {
    return;
  }
  spec_mat.set_size(spec_params->nblocks, spec_params->nfreqs);
  // TODO reuse buffers
  // TODO chunking?
  // write edf method to do diff on the fly?
  int nsamples = spec_params->nsamples;
  float* buf1 = create_buffer(nsamples);
  float* buf2 = create_buffer(nsamples);
  if (buf1 == NULL || buf2 == NULL)
  {
    return;
  }

  // nfreqs x nblocks matrix
  spec_mat.fill(0);

  int ch_idx1, ch_idx2, n;
  ch_idx1 = DIFFERENCE_PAIRS[ch].ch_idx[0];
  n = read_samples(spec_params->hdl, ch_idx1, nsamples, buf1);
  if (n == - 1)
  {
    return;
  }

  for (int i = 1; i < NUM_DIFFS; i++)
  {
    ch_idx2 = DIFFERENCE_PAIRS[ch].ch_idx[i];
    n = read_samples(spec_params->hdl, ch_idx2, nsamples, buf2);
    if (n == -1 )
    {
      return;
    }

    // TODO use rowvec::fixed with fixed size chunks
    frowvec v1 = frowvec(buf1, nsamples);
    frowvec v2 = frowvec(buf2, nsamples);
    frowvec diff = v2 - v1;

    // fill in the spec matrix with fft values
    STFT(diff, spec_params, spec_mat);
    std::swap(buf1, buf2);
  }
  // TODO serialize spec_mat output for each channel
  spec_mat /=  (NUM_DIFFS - 1); // average diff spectrograms
  spec_mat = spec_mat.t(); // transpose the output

  free(buf1);
  free(buf2);
}
/*
 * Transform the mat to a float* for transfer
 * via websockets
 */
void serialize_spec_mat(spec_params_t* spec_params, fmat& spec_mat, float* spec_arr)
{
  if (spec_params->hdl == -1)
  {
    return;
  }
  for (int i = 0; i < spec_params->nfreqs; i++)
  {
    for (int j = 0; j < spec_params->nblocks; j++)
    {
      *(spec_arr + i + j * spec_params->nfreqs) = (float) spec_mat(i, j);
    }
  }
}
