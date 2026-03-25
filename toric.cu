#include <stdio.h>
#include <assert.h>
#include <array>
#include <iostream>

using namespace std;

inline
cudaError_t checkCuda(cudaError_t result)
{
    #if defined(DEBUG) || defined(_DEBUG)
        if (result != cudaSuccess)
        {
            f//printf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
            assert(result == cudaSuccess);
        }
    #endif
    return result;
}

constexpr int L = 64;

int parent[L*L];
int edges[2*L*L];
int parity[L*L];
int ans_parity[L*L];
int output[2*L*L];

__global__ void get_syndromes(int* edges, int* parity)
{
    //__shared__ int l_p[L][L+1];
    __shared__ int l_edge[2*L][L+1];

    int idx = threadIdx.x + threadIdx.y*blockDim.x;
    int stride = blockDim.x*blockDim.y;
    for (int i = idx; i < 2*L*L; i += stride)
    {
        int x = i % L;
        int y = int(i / L);
        l_edge[y][x] = edges[i];
    }

    __syncthreads();

    for (int i = idx; i < L*L; i += stride)
    {
        int x = i % L;
        int y = int(i / L);
        int n_e[8];
        n_e[0] = x;//(x + 2*y*L);
        n_e[1] = 2*y;
        n_e[2] = (x-1 + L) % L;
        n_e[3] = 2*y;
        n_e[4] = x;
        n_e[5] = ((2*y+1 + 2*L) % (2*L));
        n_e[6] = x;
        n_e[7] = ((2*y-1 + 2*L) % (2*L));
        int n = 0;
        for (int j = 0; j < 8; j += 2)
        {
            n += l_edge[n_e[j+1]][n_e[j]];
        }

        parity[i] = n % 2;
    }

}

__global__ void grow_cluster(int* parent, int* parity, int* output)
{
    __shared__ int l_parent[L][L+1];
    __shared__ u_int8_t l_parity[L][L+1];
    __shared__ u_int8_t l_output[2*L][L+1];

    int idx = threadIdx.x + threadIdx.y*blockDim.x;
    int stride = blockDim.x*blockDim.y;
    for (int i = idx; i < L*L; i += stride)
    {
        int x = i % L;
        int y = int(i / L);
        l_parent[y][x] = parent[i];
        l_parity[y][x] = parity[i];
    }

    for (int i = idx; i < 2*L*L; i += stride)
    {
        int x = i % L;
        int y = int(i / L);
        l_output[y][x] = 0;
    }

    __syncthreads();

    __shared__ int has_odd;

    do
    {
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            has_odd = 0;
        }
        //path compression
        for (int i = idx; i < L*L; i += stride)
        {
            int x = i % L;
            int y = int(i / L);
            int parent_idx = l_parent[y][x];
            int x2 = parent_idx % L;
            int y2 = int(parent_idx / L);
            int p = l_parent[y2][x2];
            if (p != parent_idx)
            {
                parent[i] = p;
                l_parent[y][x] = p;
            }
        }

        for (int i = idx; i < L*L; i += stride)
        {
            if (l_parent[int(i / L)][i % L] == i)
            {
                if (l_parity[int(i / L)][i % L])
                {
                    atomicOr(&has_odd, 1);
                }
            }
        }

        if (!has_odd)
        {
            break;
        }

        //cluster handling
        for (int i = idx; i < L*L; i += stride)
        {
            int x = i % L;
            int y = int(i / L);
            int p = l_parent[y][x];
            int n_e[8];
            n_e[0] = (x+1 + L) % L;
            n_e[1] = y;
            n_e[2] = (x-1 + L) % L;
            n_e[3] = y;
            n_e[4] = x;
            n_e[5] = (y+1 + L) % L;
            n_e[6] = x;
            n_e[7] = (y-1 + L) % L;

            int n_edge[4];
            n_edge[0] = (x + 2*y*L);
            n_edge[1] = ((x-1 + L) % L + 2*y*L);
            n_edge[2] = (x + (2*y+1 + 2*L) % (2*L)*L);
            n_edge[3] = (x + (2*y-1 + 2*L) % (2*L)*L);

            //figure out if boundary, and expand as needed
            ////printf("%i\n", l_parity[int(p / L)][p % L]);
            if (l_parity[int(p / L)][p % L] == 0)
            {
                //printf("We're actually not a syndrome %i\n", i);
                continue;
            }
            else
            {
                //printf("I am a syndrome! %i %i\n", p, i);
            }
            //priority sweep
            for (int j = 0; j < 8; j += 2)
            {
                if (p != l_parent[y][x])
                {
                    break;
                }
                if (l_parity[int(p / L)][p % L] == 0)
                {
                    //printf("I am no longer a syndrome %d\n", i);
                    break;
                }
                int p2 = l_parent[n_e[j+1]][n_e[j]];
                if (p2 != p && l_parity[int(p2 / L)][p2 % L])
                {
                    //printf("What's going on here bruh %i %i %i (%i %i) (%i %i)\n", p, p2, i, x, y, n_e[j], n_e[j+1]);
                    int x2 = p2 % L;
                    int y2 = int(p2 / L);
                    int p3 = l_parent[y2][x2]; //parent node and parent value
                    
                    //get own parent location
                    int px = p % L;
                    int py = int(p / L);

                    if (p < p3)
                    {
                        if (p3 == atomicCAS(&l_parent[y2][x2], p3, p))
                        {
                            //printf("Atomically CASing it %d %d %d %d\n", p, p3, i, j);
                            //l_parity[py][px] ^= l_parity[y2][x2];
                            //l_parity[y2][x2] = l_parity[py][px];
                            l_parity[py][px] ^= l_parity[int(p3 / L)][p3 % L];
                            l_parity[int(p3 / L)][p3 % L] = l_parity[py][px];
                            if (p != p3)
                            {
                                l_output[int(n_edge[j/2] / L)][n_edge[j/2] % L] = 3;
                            }
                        }
                    }
                    else
                    {
                        if (p == atomicCAS(&l_parent[py][px], p, p3))
                        {
                            //printf("Atomically CASing it %d %d %d %d\n", p, p3, i, j);
                            //l_parity[y2][x2] ^= l_parity[py][px];
                            //l_parity[py][px] = l_parity[y2][x2];
                            l_parity[int(p3 / L)][p3 % L] ^= l_parity[py][px];
                            l_parity[py][px] = l_parity[int(p3 / L)][p3 % L];
                            if (p != p3)
                            {
                                l_output[int(n_edge[j/2] / L)][n_edge[j/2] % L] = 3;
                            }
                        }
                    }
                }
            }

            //full sweep
            for (int j = 0; j < 8; j += 2)
            {
                if (p != l_parent[y][x])
                {
                    break;
                }
                if (l_parity[int(p / L)][p % L] == 0)
                {
                    //printf("I am no longer a syndrome %d\n", i);
                    break;
                }
                int p2 = l_parent[n_e[j+1]][n_e[j]];
                if (p2 != p)
                {
                    //printf("What's going on here bruh %i %i %i (%i %i) (%i %i)\n", p, p2, i, x, y, n_e[j], n_e[j+1]);
                    int x2 = p2 % L;
                    int y2 = int(p2 / L);
                    int p3 = l_parent[y2][x2]; //parent node and parent value
                    
                    //get own parent location
                    int px = p % L;
                    int py = int(p / L);

                    if (p < p3)
                    {
                        if (p3 == atomicCAS(&l_parent[y2][x2], p3, p))
                        {
                            //printf("Atomically CASing it %d %d %d %d\n", p, p3, i, j);
                            //l_parity[py][px] ^= l_parity[y2][x2];
                            //l_parity[y2][x2] = l_parity[py][px];
                            l_parity[py][px] ^= l_parity[int(p3 / L)][p3 % L];
                            l_parity[int(p3 / L)][p3 % L] = l_parity[py][px];
                            //printf("What the heck %i %i %i\n", int(p3 / L), p3 % L, l_parity[int(p3 / L)][p3 % L]);
                            //printf("What the heck %i %i %i\n", py, px, l_parity[py][px]);
                            if (p != p3)
                            {
                                l_output[int(n_edge[j/2] / L)][n_edge[j/2] % L] = 3;
                            }
                        }
                    }
                    else
                    {
                        if (p == atomicCAS(&l_parent[py][px], p, p3))
                        {
                            //printf("Atomically CASing it %d %d %d %d\n", p, p3, i, j);
                            //l_parity[y2][x2] ^= l_parity[py][px];
                            //l_parity[py][px] = l_parity[y2][x2];
                            l_parity[int(p3 / L)][p3 % L] ^= l_parity[py][px];
                            l_parity[py][px] = l_parity[int(p3 / L)][p3 % L];
                            //printf("What the heck %i %i %i\n", int(p3 / L), p3 % L, l_parity[int(p3 / L)][p3 % L]);
                            //printf("What the heck %i %i %i\n", py, px, l_parity[py][px]);
                            if (p != p3)
                            {
                                l_output[int(n_edge[j/2] / L)][n_edge[j/2] % L] = 3;
                            }
                        }
                    }
                }
            }

        }

        __syncthreads();

    } while (has_odd);

    __syncthreads();

    __shared__ int max;
    for (int i = idx; i < L*L; i += stride)
    {
        int x = i % L;
        int y = int(i / L);
        l_parity[y][x] = parity[i];
    }

    if (idx == 0)
    {
        max = 10;
    }

    do
    {
        //peeling
        for (int i = idx; i < L*L; i += stride)
        {
            l_parent[int(i / L)][i % L] = 0;
        }

        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            has_odd = 0;
            max--;
        }
        __syncthreads();

        for (int i = idx; i < 2*L*L; i += stride)
        {
            if(l_output[int(i / L)][i % L] == 3)
            {
                atomicOr(&has_odd, 1);
            };
        }

        if (!has_odd || max <= 0)
        {
            //printf("%i", threadIdx.x);
            break;
        }
        else
        {
            //printf("Not quite there yet\n");
        }

        for (int i = idx; i < L*L; i += stride)
        {
            int x = i % L;
            int y = int(i / L);
            int p = l_parent[y][x];
            int n_e[8];
            n_e[0] = (x+1 + L) % L;
            n_e[1] = y;
            n_e[2] = (x-1 + L) % L;
            n_e[3] = y;
            n_e[4] = x;
            n_e[5] = (y+1 + L) % L;
            n_e[6] = x;
            n_e[7] = (y-1 + L) % L;

            int n_edge[4];
            n_edge[0] = (x + 2*y*L);
            n_edge[1] = ((x-1 + L) % L + 2*y*L);
            n_edge[2] = (x + (2*y+1 + 2*L) % (2*L)*L);
            n_edge[3] = (x + (2*y-1 + 2*L) % (2*L)*L);

            int count = 0;
            for (int j = 0; j < 4; j++)
            {
                if (l_output[int(n_edge[j] / L)][n_edge[j] % L] == 3)
                {
                    count++;
                }
            }
            if (count == 1)
            {
                for (int j = 0; j < 4; j++)
                {
                    if (l_output[int(n_edge[j] / L)][n_edge[j] % L] == 3)
                    {
                        if (0 == atomicCAS(&l_parent[n_e[2*j+1]][n_e[2*j]], 0, 1))
                        {
                            if (l_parity[y][x] == 1)
                            {
                                l_parity[y][x] = 0;
                                l_parity[n_e[2*j+1]][n_e[2*j]] ^= 1;
                                l_output[int(n_edge[j] / L)][n_edge[j] % L] = 1;
                            }
                            else
                            {
                                l_output[int(n_edge[j] / L)][n_edge[j] % L] = 0;
                            }
                        }
                    }
                }
            }
        }

        for (int i = idx; i < L*L; i += stride)
        {
            ////printf("%i %i %i\n", int(i/L), i%L, l_parity[int(i / L)][i % L]);
            parity[i] = l_parity[int(i / L)][i % L];
            parent[i] = l_parent[int(i / L)][i % L];
        }
        for (int i = idx; i < 2*L*L; i+= stride)
        {
            output[i] = l_output[int(i / L)][i % L];
        }
        __syncthreads();
    } while (has_odd);
}

void set_errors(int* qubits, int* parity, double error_rate)
{
    //set qubit errors
    /*for (int i = 0; i < 2*L*L; i++)
    {
        if (double(rand())/RAND_MAX < error_rate)
        {
            qubits[i] = 1;
        }
        else
        {
            qubits[i] = 0;
        }
    }//*/
   //qubits[13] = 1;
   
   /*qubits[11] = 1;
   qubits[18] = 1;
   qubits[19] = 1;*/

   //qubits[0] = 1;
   //qubits[6] = 1;
   //qubits[9] = 1;
   //qubits[49] = 1;

   qubits[24] = 1;
   qubits[27] = 1;
   qubits[37] = 1;
   qubits[46] = 1;


    //set syndromes
    for (int i = 0; i < L*L; i++)
    {
        //calculate neighbors
        int x = i % L;
        int y = int(i / L);
        int n_e[4];
        n_e[0] = (x + 2*y*L);
        n_e[1] = (((x-1 + L) % L) + 2*y*L);
        n_e[2] = (x + ((2*y+1 + 2*L) % (2*L))*L);
        n_e[3] = (x + ((2*y-1 + 2*L) % (2*L))*L);
        int n = 0;
        for (int j = 0; j < 4; j++)
        {
            n += qubits[n_e[j]];
        }

        
        parity[i] = n % 2;
        /*if (x == 4 && y == 4)
        {
            parity[x+y*L] = 3;
            for (int j = 0; j < 4; j++)
            {
                cout<<n_e[j]<<" "<<x<<" "<<y<<endl;
                qubits[n_e[j]] = 5;
            }
        }*/
    }

}

void init_input()
{
    for (int i = 0; i < L*L; i++)
    {
        parent[i] = i;
        parity[i] = 0;
        ans_parity[i] = 0;
    }

    set_errors(edges, ans_parity, 0.05);
}

void display(int* vertices, int* edges)
{
    for (int y = 0; y < L*2; y+=2)
    {
        for (int x = 0; x < L; x++)
        {
            char c = '-';
            if (vertices[int(y/2)*L + x])
            {
                c = 'X';
                //c = char(vertices[int(y/2)*L + x]);
            }
            cout<<c<<"--"<<edges[y*L+x]<<"--";
        }
        cout<<endl;
        for (int x = 0; x < L; x++)
        {
            cout<<"|     ";
        }
        cout<<endl;
        for (int x = 0; x < L; x++)
        {
            cout<<edges[(y+1)*L+x]<<"     ";
        }
        cout<<endl;
        for (int x = 0; x < L; x++)
        {
            cout<<"|     ";
        }
        cout<<endl;
    }
}

void display2(int* vertices, int* edges)
{
    for (int y = 0; y < L*2; y+=2)
    {
        for (int x = 0; x < L; x++)
        {
            char c = '0' + (vertices[int(y/2)*L + x] % 10);
            cout<<c<<"--"<<edges[y*L+x]<<"--";
        }
        cout<<endl;
        for (int x = 0; x < L; x++)
        {
            cout<<"|     ";
        }
        cout<<endl;
        for (int x = 0; x < L; x++)
        {
            cout<<edges[(y+1)*L+x]<<"     ";
        }
        cout<<endl;
        for (int x = 0; x < L; x++)
        {
            cout<<"|     ";
        }
        cout<<endl;
    }
}

void checkResults(int* in, int* ans, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (in[i] != ans[i])
        {
            cout<<"Error found"<<endl;
        }
    }
}


int main()
{
    srand(time(NULL));
    init_input();

    int* d_parity;
    int* d_edges;
    int* d_parent;
    int* d_output;
    int bytes = L*L*sizeof(int);
    dim3 grid(1, 1);
    dim3 block(L, 4);
    checkCuda ( cudaMalloc((void**)&d_parity, bytes));
    checkCuda ( cudaMalloc((void**)&d_edges, 2*bytes));
    checkCuda ( cudaMalloc((void**)&d_parent, bytes));
    checkCuda ( cudaMalloc((void**)&d_output, 2*bytes));
    checkCuda( cudaMemcpy(d_parity, parity, bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_edges, edges, 2*bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_parent, parent, bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_output, output, 2*bytes, cudaMemcpyHostToDevice) );  

    const int nReps = 20;
    float milliseconds;
    cudaEvent_t startEvent, stopEvent;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );

    get_syndromes<<<grid, block>>>(d_edges, d_parity);

    
    for (int i = 0; i < nReps; i++)
        get_syndromes<<<grid, block>>>(d_edges, d_parity);

    checkCuda( cudaMemcpy(parity, d_parity, bytes, cudaMemcpyDeviceToHost) );

    checkResults(parity, ans_parity, L*L);

    //printf("   Average Bandwidth (GB/s): %f\n\n", 1e-6 * bytes * nReps / milliseconds);


    display(parity, edges);
    cout<<"-------------------------------------"<<endl;
    char c = 'a';
    //while (c != 'q')
    //{
    checkCuda( cudaEventRecord(startEvent, 0) );
    grow_cluster<<<grid, block>>>(d_parent, d_parity, d_output);
    checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );

    checkCuda( cudaMemcpy(parent, d_parent, bytes, cudaMemcpyDeviceToHost) );
    checkCuda( cudaMemcpy(parity, d_parity, bytes, cudaMemcpyDeviceToHost) );
    checkCuda( cudaMemcpy(output, d_output, 2*bytes, cudaMemcpyDeviceToHost) );
    


    /*display(ans_parity, edges);
    cout<<"-------------------------------------"<<endl;
    display(parity, edges);*/
    display2(parent, output);
    cout<<"-------------------------------------"<<endl;
    display(parity, edges);
    cout<<"Milliseconds: "<<milliseconds<<endl;

    for (int i = 0; i < 2*L*L; i++)
    {
        if (output[i] == 1)
        {
            edges[i] ^= output[i];
        }
    }
    for (int i = 0; i < L*L; i++)
    {
        parity[i] = 0;
    }

    for (int i = 0; i < L*L; i++)
    {
        //calculate neighbors
        int x = i % L;
        int y = int(i / L);
        int n_e[4];
        n_e[0] = (x + 2*y*L);
        n_e[1] = (((x-1 + L) % L) + 2*y*L);
        n_e[2] = (x + ((2*y+1 + 2*L) % (2*L))*L);
        n_e[3] = (x + ((2*y-1 + 2*L) % (2*L))*L);
        int n = 0;
        for (int j = 0; j < 4; j++)
        {
            n += edges[n_e[j]];
        }

        
        if (n%2)
        {
            cout<<"Ruh roh scoobs"<<i<<endl;
        }
        parity[i] = n%2;
    }
    //cout<<"Actual output"<<endl;
    //display(parity, edges);
        //cin>>c;
    //}
}