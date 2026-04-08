#include <stdio.h>
#include <assert.h>
#include <array>
#include <iostream>

#define mask 0x7FFFFFFF

using namespace std;

inline
cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
        {
            printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
            assert(result == cudaSuccess);
        }
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
    __shared__ unsigned int l_parent[L][L+1];
    __shared__ int locks[L][L+1];
    __shared__ u_int8_t l_parity[L][L+1];
    __shared__ u_int8_t l_output[2*L][L+1];

    int idx = threadIdx.x + threadIdx.y*blockDim.x;
    int stride = blockDim.x*blockDim.y;
    for (int i = idx; i < L*L; i += stride)
    {
        int x = i % L;
        int y = int(i / L);
        l_parent[y][x] = parent[i] | (parity[i] << 31);
        locks[y][x] = 0;
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
            int parent_idx = l_parent[y][x] & mask;
            int p = parent_idx;
            do
            {
            
                int x2 = p % L;
                int y2 = int(p / L);
                p = l_parent[y2][x2] & mask;
                if (p == y2*L + x2)
                {
                    p = l_parent[y2][x2];
                    break;
                }
            } while (true);
            l_parent[y][x] = p;
            locks[y][x] = 0;
        }
        __syncthreads();
        
        //cluster handling
        for (int i = idx; i < L*L; i += stride)
        {
            int x = i % L;
            int y = int(i / L);
            int p = l_parent[y][x] & mask;
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
            if ((l_parent[int(p / L)][p % L] >> 31) == 0)
            {
                //continue;
            }
            else
            {
                //priority sweep
                for (int j = 0; j < 8; j += 2)
                {
                    unsigned int parity1 = (l_parent[int(p / L)][p % L] >> 31);
                    int p2 = l_parent[n_e[j+1]][n_e[j]] & mask;
                    if (p2 != p && (l_parent[int(p2 / L)][p2 % L] >> 31))
                    {
                        int x2 = p2 % L;
                        int y2 = int(p2 / L);
                        unsigned int p3 = l_parent[y2][x2]; //parent node and parent value
                        p3 = p3 & mask;
                        x2 = p3 % L;
                        y2 = int(p3 / L);

                        unsigned int parity3 = l_parent[int(p3 / L)][p3 % L] >> 31;
                        
                        //get own parent location
                        int px = p % L;
                        int py = int(p / L);

                        int small = min(p, p3);
                        int big = max(p, p3);
                        if (atomicCAS(&locks[int(small/L)][small % L], 0, 1) == 0)
                        {
                            if (atomicCAS(&locks[int(big/L)][big % L], 0, 1) == 0)
                            {
                                parity1 = l_parent[int(p/L)][p%L]>>31;
                                parity3 = l_parent[int(p3/L)][p3%L]>>31;
                                y2 = int(p3/L);
                                x2 = p3%L;
                                py = int(p/L);
                                px = p%L;
                                

                                if (p < p3)
                                {
                                    //if parent is its own parent
                                    if ((p3 | (parity3<<31)) == atomicCAS(&l_parent[y2][x2], p3 | (parity3<<31), p | ((parity3 ^ parity1)<<31)))
                                    {
                                        l_parent[py][px] = l_parent[y2][x2];
                                        if (p != p3 && p != p2)
                                        {
                                            l_output[int(n_edge[j/2] / L)][n_edge[j/2] % L] = 3;
                                            atomicOr(&has_odd, 1);
                                        }
                                    }
                                }
                                else
                                {
                                    if ((p | (parity1<<31)) == atomicCAS(&l_parent[py][px], p | (parity1<<31), p3 | ((parity3 ^ parity1)<<31)))
                                    {
                                        l_parent[y2][x2] = l_parent[py][px];
                                        if (p != p3 && p != p2)
                                        {
                                            l_output[int(n_edge[j/2] / L)][n_edge[j/2] % L] = 3;
                                            atomicOr(&has_odd, 1);
                                        }
                                    }
                                }
                            }
                            locks[int(small/L)][small % L] = 0;
                        }
                    }
                }

                //full sweep
                for (int j = 0; j < 8; j += 2)
                {
                    unsigned int parity1 = (l_parent[int(p / L)][p % L] >> 31);
                    int p2 = l_parent[n_e[j+1]][n_e[j]] & mask;
                    if (p2 != p)
                    {
                        int x2 = p2 % L;
                        int y2 = int(p2 / L);
                        unsigned int p3 = l_parent[y2][x2]; //parent node and parent value
                        p3 = p3 & mask;
                        x2 = p3 % L;
                        y2 = int(p3 / L);
                        unsigned int parity3 = l_parent[int(p3 / L)][p3 % L] >> 31;
                        
                        //get own parent location
                        int px = p % L;
                        int py = int(p / L);

                        int small = min(p, p3);
                        int big = max(p, p3);
                        if (atomicCAS(&locks[int(small/L)][small % L], 0, 1) == 0)
                        {
                            if (atomicCAS(&locks[int(big/L)][big % L], 0, 1) == 0)
                            {
                                parity1 = l_parent[int(p/L)][p%L]>>31;
                                parity3 = l_parent[int(p3/L)][p3%L]>>31;
                                y2 = int(p3/L);
                                x2 = p3%L;
                                py = int(p/L);
                                px = p%L;

                                if (p < p3)
                                {
                                    //if parent is its own parent
                                    if ((p3 | (parity3<<31)) == atomicCAS(&l_parent[y2][x2], p3 | (parity3<<31), p | ((parity3 ^ parity1)<<31)))
                                    {
                                        l_parent[py][px] = l_parent[y2][x2];
                                        if (p != p3 && p != p2)
                                        {
                                            l_output[int(n_edge[j/2] / L)][n_edge[j/2] % L] = 3;
                                            atomicOr(&has_odd, 1);
                                        }
                                    }
                                }
                                else
                                {
                                    if ((p | (parity1<<31)) == atomicCAS(&l_parent[py][px], p | (parity1<<31), p3 | ((parity3 ^ parity1)<<31)))
                                    {
                                        l_parent[y2][x2] = l_parent[py][px];
                                        if (p != p3 && p != p2)
                                        {
                                            l_output[int(n_edge[j/2] / L)][n_edge[j/2] % L] = 3;
                                            atomicOr(&has_odd, 1);
                                        }
                                    }
                                }
                            }
                            locks[int(small/L)][small % L] = 0;
                        }

                    }
                }
            }

            __syncthreads();
        }

        __syncthreads();

    } while (has_odd);

    __syncthreads();

    for (int i = idx; i < L*L; i += stride)
    {
        int x = i % L;
        int y = int(i / L);
        l_parity[y][x] = parity[i];
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
        }
        __syncthreads();

        for (int i = idx; i < 2*L*L; i += stride)
        {
            if(l_output[int(i / L)][i % L] == 3)
            {
                atomicOr(&has_odd, 1);
            };
        }

        __syncthreads();

        if (!has_odd)
        {
            break;
        }

        for (int i = idx; i < L*L; i += stride)
        {
            int x = i % L;
            int y = int(i / L);
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
                        int i1 = y*L + x;
                        int i2 = n_e[2*j+1]*L + n_e[2*j];
                        int small = min(i1, i2);
                        int big = max(i1, i2);
                        if (0 == atomicCAS(&l_parent[int(small / L)][small % L], 0, 1))
                        {
                            if (0 == atomicCAS(&l_parent[int(big / L)][big % L], 0, 1))
                            {
                                if (l_parity[y][x])
                                {
                                    l_parity[y][x] ^= 1;
                                    l_parity[n_e[2*j+1]][n_e[2*j]] ^= 1;
                                    l_output[int(n_edge[j] / L)][n_edge[j] % L] = 1;
                                }
                                else
                                {
                                    l_output[int(n_edge[j] / L)][n_edge[j] % L] = 0;
                                }
                                l_parent[n_e[2*j+1]][n_e[2*j]] = 0;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        
    } while (has_odd);

    //__syncthreads();

    for (int i = idx; i < L*L; i += stride)
    {
        parity[i] = l_parity[int(i / L)][i % L];
        parent[i] = l_parent[int(i / L)][i % L] & mask;
    }
    for (int i = idx; i < 2*L*L; i+= stride)
    {
        output[i] = l_output[int(i / L)][i % L];
    }
}

void set_errors(int* qubits, int* parity, double error_rate)
{
    //set qubit errors
    for (int i = 0; i < 2*L*L; i++)
    {
        if (double(rand())/RAND_MAX < error_rate)
        {
            qubits[i] = 1;
        }
        else
        {
            qubits[i] = 0;
        }
    }


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
    int iter = 500;
    float t_millis = 0;
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
    for (int k = 0; k < iter; k++)
    {
        unsigned int seed = time(NULL) + k;
        //unsigned int seed = 1775658110;
        //cout<<"Seed: "<<seed<<endl;
        srand(seed);
        init_input();

        float milliseconds;
        cudaEvent_t startEvent, stopEvent;
        checkCuda( cudaEventCreate(&startEvent) );
        checkCuda( cudaEventCreate(&stopEvent) );
        checkCuda( cudaMemcpy(d_parity, parity, bytes, cudaMemcpyHostToDevice) );
        checkCuda( cudaMemcpy(d_edges, edges, 2*bytes, cudaMemcpyHostToDevice) );
        checkCuda( cudaMemcpy(d_parent, parent, bytes, cudaMemcpyHostToDevice) );
        checkCuda( cudaMemcpy(d_output, output, 2*bytes, cudaMemcpyHostToDevice) );  

        get_syndromes<<<grid, block>>>(d_edges, d_parity);

        checkCuda( cudaMemcpy(parity, d_parity, bytes, cudaMemcpyDeviceToHost) );

        checkResults(parity, ans_parity, L*L);


        //display(parity, edges);
        checkCuda( cudaEventRecord(startEvent, 0) );
        grow_cluster<<<grid, block>>>(d_parent, d_parity, d_output);
        checkCuda( cudaDeviceSynchronize() );
        checkCuda( cudaEventRecord(stopEvent, 0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );

        checkCuda( cudaMemcpy(parent, d_parent, bytes, cudaMemcpyDeviceToHost) );
        checkCuda( cudaMemcpy(parity, d_parity, bytes, cudaMemcpyDeviceToHost) );
        checkCuda( cudaMemcpy(output, d_output, 2*bytes, cudaMemcpyDeviceToHost) );
        


        //uncomment for debug output
        /*display(ans_parity, edges);
        cout<<"-------------------------------------"<<endl;
        display(parity, edges);
        display2(parent, output);
        cout<<"-------------------------------------";*/
        for (int i = 0; i < 2*L*L; i++)
        {
            if (output[i] == 1)
            {
                edges[i] ^= output[i];
            }
        }
        //display(parity, edges);
        //cout<<"Milliseconds: "<<milliseconds<<" seed: "<<seed<<endl;
        t_millis += milliseconds;
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
                cout<<"Issue in output found: "<<i<<" "<<k<<" "<<milliseconds<<" "<<seed<<endl;
            }
            parity[i] = n%2;
        }
        
        //cleanup
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }
    cudaFree(d_parity);
    cudaFree(d_edges);
    cudaFree(d_parent);
    cudaFree(d_output);
    cout<<"Lattice size: "<<L<<"x"<<L<<endl;
    cout<<"Number of runs: "<<iter<<endl;
    cout<<"Milliseconds per lattice: "<<t_millis/iter<<endl;
    //display(parity, edges);
        //cin>>c;
    //}
}