#include "permutohedral.h"

#include <cmath>

#include <unordered_map>
#include <vector>


using std::unordered_map;
using std::vector;

/**
 * (c) Philip Krahenbuhel.
 * NOTE: it works faster than standard containers
 */
class HashTable
{
 protected:
  size_t key_size_, filled_, capacity_;
  std::vector< short > keys_;
  std::vector< int > table_;
  void grow(){
    // Create the new memory and copy the values in
    int old_capacity = capacity_;
    capacity_ *= 2;
    std::vector<short> old_keys( (old_capacity+10)*key_size_ );
    std::copy( keys_.begin(), keys_.end(), old_keys.begin() );
    std::vector<int> old_table( capacity_, -1 );

    // Swap the memory
    table_.swap( old_table );
    keys_.swap( old_keys );

    // Reinsert each element
    for( int i=0; i<old_capacity; i++ )
      if (old_table[i] >= 0){
        int e = old_table[i];
        size_t h = hash( getKey(e) ) % capacity_;
        for(; table_[h] >= 0; h = h<capacity_-1 ? h+1 : 0);
        table_[h] = e;
      }
  }
  size_t hash(const short * k) {
    size_t r = 0;
    for (size_t i = 0; i < key_size_; i++) {
      r += k[i];
      r *= 1664525;
    }
    return r;
  }
 public:
  explicit HashTable( int key_size, int n_elements ) :
      key_size_ ( key_size ), filled_(0), capacity_(2*n_elements),
      keys_((capacity_/2+10)*key_size_), table_(2*n_elements,-1) {
  }

  int size() const {
    return filled_;
  }

  void reset() {
    filled_ = 0;
    std::fill( table_.begin(), table_.end(), -1 );
  }

  int find(const short * k, bool create = false) {
    if (2*filled_ >= capacity_) grow();
    // Get the hash value
    size_t h = hash( k ) % capacity_;
    // Find the element with he right key, using linear probing
    while (1) {
      int e = table_[h];
      if (e==-1){
        if (create){
          // Insert a new key and return the new id
          for( size_t i=0; i<key_size_; i++ )
            keys_[ filled_*key_size_+i ] = k[i];
          return table_[h] = filled_++;
        }
        else
          return -1;
      }
      // Check if the current key is The One
      bool good = true;
      for( size_t i=0; i<key_size_ && good; i++ )
        if (keys_[ e*key_size_+i ] != k[i])
          good = false;
      if (good)
        return e;
      // Continue searching
      h++;
      if (h==capacity_) h = 0;
    }
  }

  const short * getKey(int i) const {
    return &keys_[i*key_size_];
  }

};

namespace std {
template<> struct hash<std::vector<short>> {
  typedef std::vector<short> argument_type;
  typedef std::size_t result_type;
  result_type operator()(argument_type const& k) const {
    std::size_t r = 0;
    for (size_t i = 0; i < k.size(); i++) {
      r += k[i];
      r *= 1664525;
    }
    return r;
  }
};
}



namespace permutohedral {

void init_sse(const float* features, int n_points, int n_features,
              int32* offsets, float* weights,
              int32*& neighbours, int &n_vertices)
{
  int N = n_points;
  int F = n_features;

  // Compute the lattice coordinates for each feature [there is going to be a lot of magic here
  HashTable hash_table( F, N/**(F+1)*/ );

  const int blocksize = sizeof(__m128) / sizeof(float);
  const __m128 invdplus1   = _mm_set1_ps( 1.0f / (F+1) );
  const __m128 dplus1      = _mm_set1_ps( F+1 );
  const __m128 Zero        = _mm_set1_ps( 0 );
  const __m128 One         = _mm_set1_ps( 1 );

  // Allocate the class memory
  // TODO: this should be done in the op initialization
  // offset_.resize( (F+1)*(N_+16) );
  // std::fill( offset_.begin(), offset_.end(), 0 );
  // barycentric_.resize( (F+1)*(N_+16) );
  // std::fill( barycentric_.begin(), barycentric_.end(), 0 );
  // rank_.resize( (F+1)*(N_+16) );

  // Allocate the local memory
  __m128 * scale_factor = (__m128*) _mm_malloc( (F  )*sizeof(__m128) , 16 );
  __m128 * f            = (__m128*) _mm_malloc( (F  )*sizeof(__m128) , 16 );
  __m128 * elevated     = (__m128*) _mm_malloc( (F+1)*sizeof(__m128) , 16 );
  __m128 * rem0         = (__m128*) _mm_malloc( (F+1)*sizeof(__m128) , 16 );
  __m128 * rank         = (__m128*) _mm_malloc( (F+1)*sizeof(__m128), 16 );
  float * barycentric = new float[(F+2)*blocksize];
  short * canonical = new short[(F+1)*(F+1)];
  short * key = new short[F+1];

  // Compute the canonical simplex
  for( int i=0; i<=F; i++ ){
    for( int j=0; j <= F-i; j++ )
      canonical[i*(F+1)+j] = i;
    for( int j=F-i+1; j<=F; j++ )
      canonical[i*(F+1)+j] = i - (F+1);
  }

  // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
  float inv_std_dev = sqrt(2.0 / 3.0)*(F+1);
  // Compute the diagonal part of E (p.5 in [Adams etal 2010])
  for( int i=0; i<F; i++ )
    scale_factor[i] = _mm_set1_ps( 1.0 / sqrt( (i+2)*(i+1) ) * inv_std_dev );

  // Setup the SSE rounding
#ifndef __SSE4_1__
  const unsigned int old_rounding = _mm_getcsr();
  _mm_setcsr( (old_rounding&~_MM_ROUND_MASK) | _MM_ROUND_NEAREST );
#endif

  // Compute the simplex each feature lies in
  for (int k = 0; k < N; k += blocksize) {

    // Load the feature from memory
    float* ff = (float*) f;
    for (int j = 0; j < F; j++)
      for (int i = 0; i < blocksize; i++)
        ff[j*blocksize+i] = k+i < N ? features[(k+i)*F + j] : 0.0;

    // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])

    // sm contains the sum of 1..n of our faeture vector
    __m128 sm = Zero;
    for (int j = F; j > 0; j--) {
      __m128 cf = f[j-1]*scale_factor[j-1];
      elevated[j] = sm - _mm_set1_ps(j)*cf;
      sm += cf;
    }
    elevated[0] = sm;

    // Find the closest 0-colored simplex through rounding
    __m128 sum = Zero;
    for( int i=0; i<=F; i++ ){
      __m128 v = invdplus1 * elevated[i];
#ifdef __SSE4_1__
      v = _mm_round_ps( v, _MM_FROUND_TO_NEAREST_INT );
#else
      v = _mm_cvtepi32_ps( _mm_cvtps_epi32( v ) );
#endif
      rem0[i] = v*dplus1;
      sum += v;
    }

    // Find the simplex we are in and store it in rank
    // (where rank describes what position coorinate i has in the sorted order of the features values)
    for( int i=0; i<=F; i++ )
      rank[i] = Zero;
    for( int i=0; i<F; i++ ){
      __m128 di = elevated[i] - rem0[i];
      for( int j=i+1; j<=F; j++ ){
        __m128 dj = elevated[j] - rem0[j];
        __m128 c = _mm_and_ps( One, _mm_cmplt_ps( di, dj ) );
        rank[i] += c;
        rank[j] += One-c;
      }
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for( int i=0; i<=F; i++ ){
      rank[i] += sum;
      __m128 add = _mm_and_ps( dplus1, _mm_cmplt_ps( rank[i], Zero ) );
      __m128 sub = _mm_and_ps( dplus1, _mm_cmpge_ps( rank[i], dplus1 ) );
      rank[i] += add-sub;
      rem0[i] += add-sub;
    }

    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for( int i=0; i<(F+2)*blocksize; i++ )
      barycentric[ i ] = 0;
    for( int i=0; i<=F; i++ ){
      __m128 v = (elevated[i] - rem0[i])*invdplus1;

      // Didn't figure out how to SSE this
      float * fv = (float*)&v;
      float * frank = (float*)&rank[i];
      for( int j=0; j<blocksize; j++ ){
        int p = F-frank[j];
        barycentric[j*(F+2)+p  ] += fv[j];
        barycentric[j*(F+2)+p+1] -= fv[j];
      }
    }

    // The rest is not SSE'd
    for( int j=0; j<blocksize; j++ ){
      // Wrap around
      barycentric[j*(F+2)+0]+= 1 + barycentric[j*(F+2)+F+1];

      float * frank = (float*)rank;
      float * frem0 = (float*)rem0;
      // Compute all vertices and their offset
      for( int remainder=0; remainder<=F; remainder++ ){
        for( int i=0; i<F; i++ ){
          key[i] = frem0[i*blocksize+j] + canonical[ remainder*(F+1) + (int)frank[i*blocksize+j] ];
        }
        offsets[ (j+k)*(F+1)+remainder ] = hash_table.find( key, true );
        weights[ (j+k)*(F+1)+remainder ] = barycentric[ j*(F+2)+remainder ];
      }
    }
  }
  _mm_free( scale_factor );
  _mm_free( f );
  _mm_free( elevated );
  _mm_free( rem0 );
  _mm_free( rank );
  delete [] barycentric;
  delete [] canonical;
  delete [] key;

  // Reset the SSE rounding
#ifndef __SSE4_1__
  _mm_setcsr( old_rounding );
#endif

  // This is normally fast enough so no SSE needed here
  // Find the Neighbors of each lattice point
  // TODO: make a separate function for this?

  int M = hash_table.size();

  n_vertices = M;

  // Create the neighborhood structure
  neighbours = new int[(F+1)*M*2];
  //blur_neighbors_.resize( (F+1)*M_ );

  short* n1_key = new short[F+1];
  short* n2_key = new short[F+1];

  // For each of d+1 axes,
  for (int j = 0; j <= F; j++) {
    for (int i = 0; i < M; i++) {

      const short *key = hash_table.getKey(i);

      // getting the neighbours (???)
      for (int k = 0; k < F; k++) {
        n1_key[k] = key[k] - 1;
        n2_key[k] = key[k] + 1;
      }
      n1_key[j] = key[j] + F;
      n2_key[j] = key[j] - F;

      neighbours[(i*(F+1)+j)*2+0] = hash_table.find(n1_key);
      neighbours[(i*(F+1)+j)*2+1] = hash_table.find(n2_key);
    }
  }
  delete [] n1_key;
  delete [] n2_key;
}


void compute_sse(const float* input,
                 const int32* offsets, const float* weights, const int32* neighbours,
                 int n_values, int n_points, int n_features, int n_vertices,
                 float* output,
                 bool reverse,
                 bool add)
{
  int V = n_values;
  int N = n_points;
  int F = n_features;
  int M = n_vertices;

  const int sse_V = (V-1)*sizeof(float) / sizeof(__m128) + 1;
  // Shift all values by 1 such that -1 -> 0 (used for blurring)
  __m128 * sse_val    = (__m128*) _mm_malloc( sse_V*sizeof(__m128), 16 );
  __m128 * values     = (__m128*) _mm_malloc( (M+2)*sse_V*sizeof(__m128), 16 );
  __m128 * new_values = (__m128*) _mm_malloc( (M+2)*sse_V*sizeof(__m128), 16 );

  __m128 Zero = _mm_set1_ps( 0 );

  for( int i=0; i<(M+2)*sse_V; i++ )
    values[i] = new_values[i] = Zero;
  for( int i=0; i<sse_V; i++ )
    sse_val[i] = Zero;

  float* sdp_temp = new float[V];

  // Splatting
  for (int i = 0; i < N; i++ ){
    for (int s = 0; s < V; s++) {
      //sdp_temp[s] = input[s*N + i];
      sdp_temp[s] = input[i*V+s];
    }
    memcpy(sse_val, sdp_temp, V*sizeof(float));

    for( int j=0; j<=F; j++ ){
      int o = offsets[i*(F+1)+j]+1;
      __m128 w = _mm_set1_ps(weights[i*(F+1)+j] );
      for( int k=0; k<sse_V; k++ )
        values[ o*sse_V+k ] += w * sse_val[k];
    }
  }

  // blurring
  __m128 half = _mm_set1_ps(0.5);
  for (int j = reverse ? F : 0; j <= F && j >= 0; reverse? j-- : j++) {
    for (int i = 0; i < M; i++) {
      __m128 * old_val = values + (i+1)*sse_V;
      __m128 * new_val = new_values + (i+1)*sse_V;

      int n1 = neighbours[(i*(F+1)+j)*2]+1;
      int n2 = neighbours[(i*(F+1)+j)*2+1]+1;

      __m128 * n1_val = values + n1 * sse_V;
      __m128 * n2_val = values + n2 * sse_V;
      for (int k = 0; k < sse_V; k++)
        new_val[k] = old_val[k] + half * (n1_val[k] + n2_val[k]);
    }
    std::swap(values, new_values);
  }

  // Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
  float alpha = 1.0f / (1+powf(2, -F));

  // Slicing
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < sse_V; k++)
      sse_val[k] = Zero;
    for (int j = 0; j <= F; j++) {
      int o = offsets[i*(F+1)+j]+1;
      __m128 w = _mm_set1_ps(weights[i*(F+1)+j] * alpha);
      for( int k=0; k<sse_V; k++ )
        sse_val[k] += w * values[o*sse_V+k];
    }

    memcpy(sdp_temp, sse_val, V*sizeof(float) );
    if (!add) {
      for (int s = 0; s < V; s++) {
        //output[i + s*N] = sdp_temp[s];
        output[i*V+s] = sdp_temp[s];
      }
    } else {
      for (int s = 0; s < V; s++) {
        output[i*V+s] += sdp_temp[s];
        //output[i + s*N] += sdp_temp[s];
      }
    }
  }

  _mm_free( sse_val );
  _mm_free( values );
  _mm_free( new_values );
  delete [] sdp_temp;
}

// TODO: what we CAN do, is set the max size to make some of the stuff static
void init(const float* features, int n_points, int n_features,
          int32* offsets, float* weights,
          int32*& neighbours, int &n_vertices) {

  // LOG(INFO) << "non-SSE permutohedral init";
  // Compute the lattice coordinates for each feature
  // [there is going to be a lot of magic here
  int N_ = n_points;
  int d_ = n_features;

  HashTable hash_table( d_, N_*(d_+1) );

  // Allocate the local memory
  float * scale_factor = new float[d_];
  float * elevated = new float[d_+1];
  float * rem0 = new float[d_+1];
  float * barycentric = new float[d_+2];
  short * rank = new short[d_+1];
  short * canonical = new short[(d_+1)*(d_+1)];
  short * key = new short[d_+1];

  // Compute the canonical simplex
  for (int i = 0; i <= d_; i++) {
    for (int j = 0; j <= d_-i; j++)
      canonical[i*(d_+1)+j] = i;
    for (int j = d_-i+1; j <= d_; j++)
      canonical[i*(d_+1)+j] = i - (d_+1);
  }

  // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
  float inv_std_dev = sqrt(2.0 / 3.0)*(d_+1);
  // Compute the diagonal part of E (p.5 in [Adams etal 2010])
  for (int i = 0; i < d_; i++)
    scale_factor[i] = 1.0 / sqrt( double((i+2.0)*(i+1.0)) ) * inv_std_dev;

  // Compute the simplex each feature lies in
  for (int k = 0; k < N_; k++) {
    // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
    const float* f = (features + k * d_);

    // sm contains the sum of 1..n of our feature vector
    float sm = 0;
    for (int j = d_; j > 0; j--) {
      float cf = f[j-1]*scale_factor[j-1];
      elevated[j] = sm - j*cf;
      sm += cf;
    }
    elevated[0] = sm;

    // Find the closest 0-colored simplex through rounding
    float down_factor = 1.0f / (d_+1);
    float up_factor = (d_+1);
    int sum = 0;
    for (int i = 0; i <= d_; i++) {
      //int rd1 = round( down_factor * elevated[i]);
      int rd2;
      float v = down_factor * elevated[i];
      float up = ceilf(v) * up_factor;
      float down = floorf(v)*up_factor;
      if (up - elevated[i] < elevated[i] - down) rd2 = (short)up;
      else rd2 = (short)down;

      //if(rd1!=rd2)
      //  break;

      rem0[i] = rd2;
      sum += rd2*down_factor;
    }

    // Find the simplex we are in and store it in rank
    // (where rank describes what position coorinate
    // i has in the sorted order of the features values)
    for (int i = 0; i <= d_; i++)
      rank[i] = 0;
    for (int i = 0; i < d_; i++) {
      double di = elevated[i] - rem0[i];
      for (int j = i+1; j <= d_; j++)
        if ( di < elevated[j] - rem0[j])
          rank[i]++;
        else
          rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= d_; i++) {
      rank[i] += sum;
      if (rank[i] < 0) {
        rank[i] += d_+1;
        rem0[i] += d_+1;
      }
      else if (rank[i] > d_) {
        rank[i] -= d_+1;
        rem0[i] -= d_+1;
      }
    }

    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for (int i = 0; i <= d_+1; i++)
      barycentric[i] = 0;
    for (int i = 0; i <= d_; i++) {
      float v = (elevated[i] - rem0[i])*down_factor;
      barycentric[d_-rank[i]  ] += v;
      barycentric[d_-rank[i]+1] -= v;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[d_+1];

    // Compute all vertices and their offset
    for (int remainder = 0; remainder <= d_; remainder++) {
      for (int i = 0; i < d_; i++)
        key[i] = rem0[i] + canonical[ remainder*(d_+1) + rank[i] ];
      offsets[ k*(d_+1)+remainder ] = hash_table.find(key, true);
      weights[ k*(d_+1)+remainder ] = barycentric[ remainder ];
    }
  }
  delete [] scale_factor;
  delete [] elevated;
  delete [] rem0;
  delete [] barycentric;
  delete [] rank;
  delete [] canonical;
  delete [] key;

  // Find the Neighbors of each lattice point
  // Get the number of vertices in the lattice

  int M_ = hash_table.size();

  n_vertices = M_;

  // Create the neighborhood structure
  neighbours = new int[(d_+1)*M_*2];
  //blur_neighbors_.resize( (d_+1)*M_ );

  short* n1_key = new short[d_+1];
  short* n2_key = new short[d_+1];

  // For each of d+1 axes,
  for (int j = 0; j <= d_; j++) {
    for (int i = 0; i < M_; i++) {

      const short *key = hash_table.getKey(i);

      // getting the neighbours (???)
      for (int k = 0; k < d_; k++) {
        n1_key[k] = key[k] - 1;
        n2_key[k] = key[k] + 1;
      }
      n1_key[j] = key[j] + d_;
      n2_key[j] = key[j] - d_;

      neighbours[(i*(d_+1)+j)*2+0] = hash_table.find(n1_key);
      neighbours[(i*(d_+1)+j)*2+1] = hash_table.find(n2_key);
    }
  }
  delete [] n1_key;
  delete [] n2_key;
}


void compute(const float* input,
             const int32* offsets, const float* weights, const int32* neighbours,
             int num_values, int num_points, int num_features, int num_vertices,
             float* output,
             bool reverse,
             bool add) {
  // pass
  int V = num_values;
  int N = num_points;
  int F = num_features;
  int M = num_vertices;

  // Shift all values by 1 such that -1 -> 0 (used for blurring)
  float* values = new float[(M+2) * V];
  float* new_values = new float[(M+2) * V];

  for (int i = 0; i < (M+2)*V; i++)
    values[i] = new_values[i] = 0;

  // splatting
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= F; j++) {
      int o = offsets[i*(F+1)+j]+1;
      float w = weights[i*(F+1)+j];
      for (int k=0; k < V; k++)
        values[o*V+k] += w * input[i*V+k];
    }
  }

  // blurring
  for (int j = reverse ? F : 0; j <= F && j >= 0; reverse ? j-- : j++) {
    for (int i = 0; i < M; i++) {
      float* old_val = values + (i+1) * V;
      float* new_val = new_values + (i+1) * V;

      int n1 = neighbours[(i*(F+1)+j)*2]+1;
      int n2 = neighbours[(i*(F+1)+j)*2+1]+1;
      float* n1_val = values + n1 * V;
      float* n2_val = values + n2 * V;
      for (int k = 0; k < V; k++)
        new_val[k] = old_val[k] + 0.5 * (n1_val[k] + n2_val[k]);
    }
    std::swap(values, new_values);
  }

  // Alpha is a magic scaling constant
  // (write Andrew if you really wanna understand this)
  float alpha = 1.0f / (1 + powf(2, -F));

  // slicing
  for (int i = 0; i < N; i++) {
    if (!add) {
      for (int k = 0; k < V; k++)
        output[i*V+k] = 0;
    }
    for (int j = 0; j <= F; j++) {
      int o = offsets[i*(F+1)+j]+1;
      float w = weights[i*(F+1)+j];
      for (int k = 0; k < V; k++)
        output[i*V+k] += w * values[o*V+k] * alpha;
    }
  }

  delete [] values;
  delete [] new_values;
}

} // permutohedral
