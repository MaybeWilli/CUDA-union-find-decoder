#include "toric.h"

void ToricSimulator::display(int* vertices, int* edges)
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
            //cout<<c<<"--"<<edges[y*L+x]<<"--";
        }
        //cout<<endl;
        for (int x = 0; x < L; x++)
        {
            //cout<<"|     ";
        }
        //cout<<endl;
        for (int x = 0; x < L; x++)
        {
            //cout<<edges[(y+1)*L+x]<<"     ";
        }
        //cout<<endl;
        for (int x = 0; x < L; x++)
        {
            //cout<<"|     ";
        }
        //cout<<endl;
    }
}

void ToricSimulator::display2(int* vertices, int* edges)
{
    for (int y = 0; y < L*2; y+=2)
    {
        for (int x = 0; x < L; x++)
        {
            char c = '0' + (vertices[int(y/2)*L + x] % 10);
            //cout<<c<<"--"<<edges[y*L+x]<<"--";
        }
        //cout<<endl;
        for (int x = 0; x < L; x++)
        {
            //cout<<"|     ";
        }
        //cout<<endl;
        for (int x = 0; x < L; x++)
        {
            //cout<<edges[(y+1)*L+x]<<"     ";
        }
        //cout<<endl;
        for (int x = 0; x < L; x++)
        {
            //cout<<"|     ";
        }
        //cout<<endl;
    }
}

ToricSimulator::ToricSimulator()
{
    for (int i = 0; i < 2*L*L; i++)
    {
        edges[i] = 0;
        output[i] = 0;
    }
    for (int i = 0; i < L*L; i++)
    {
        parent[i] = i;
        size[i] = 1;
    }
    add_error();
    get_syndromes();
}

void ToricSimulator::add_error()
{
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

    /*for (int i = 0; i < 2*L*L; i++)
    {
        qubits[i] = 0;
    }
    qubits[9] = 1;
    qubits[44] = 1;*/
}

void ToricSimulator::get_syndromes()
{
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
        if (parity[i] == 1)
        {
            is_syndrome[i] = 1;
            clusters.insert(i);
            boundaries[i] = {i};
        }
        else
        {
            is_syndrome[i] = 0;
        }
        /*if (x == 4 && y == 4)
        {
            parity[x+y*L] = 3;
            for (int j = 0; j < 4; j++)
            {
                //cout<<n_e[j]<<" "<<x<<" "<<y<<endl;
                qubits[n_e[j]] = 5;
            }
        }*/
    }
}

int ToricSimulator::find(int i)
{
    int node = i;
    while (parent[node] != node)
    {
        node = parent[node];
    }

    int p = node;
    node = i;

    while (parent[node] != node)
    {
        int temp = parent[node];
        parent[node] = p;
        node = temp;
    }

    return p;
}

void ToricSimulator::union_func(int i, int j)
{
    int i_f = find(i);
    int j_f = find(j);
    if (i_f == j_f)
    {
        ////cout<<i_f<<" "<<j_f<<" "<<i<<" "<<j<<endl;
        return;
    }
    if (size[i_f] < size[j_f])
    {
        int temp = i;
        i = j;
        j = temp;
    }
    int g1 = find(i);
    int g2 = find(j);

    if (boundaries.find(g2) != boundaries.end())
    {
        if (!add_boundary.count(g1))
        {
            add_boundary[g1] = vector<int>();
        }
        add_boundary[g1].insert(add_boundary[g1].end(), add_boundary[g2].begin(), add_boundary[g2].end());
        //boundaries.erase(g2);
        remove_boundary.insert(g2);
    }

    size[g1] += size[g2];
    //cout<<"What's going on"<<parity[g1]<<parity[g2]<<endl;
    parity[g1] ^= parity[g2];
    parity[g2] = parity[g1];
    //cout<<"What "<<parity[g1]<<parity[g2]<<endl;
    parent[g2] = g1;
    remove_cluster.insert(g2);
    remove_cluster.insert(j);
    //cout<<"Huh"<<endl;

    
}

bool ToricSimulator::iterate()
{
    bool updated = false;
    remove_cluster.clear();
    remove_boundary.clear();
    add_boundary.clear();
    for (int cluster : clusters)
    {
        ////cout<<cluster<<endl;
        if (parity[find(cluster)] == 1 && remove_cluster.count(cluster) == 0)
        {
            for (int v : boundaries[find(cluster)])
            {
                int x = v % L;
                int y = int(v / L);
                int n_e[4];
                n_e[0] = (x + 2*y*L);
                n_e[1] = (((x-1 + L) % L) + 2*y*L);
                n_e[2] = (x + ((2*y+1 + 2*L) % (2*L))*L);
                n_e[3] = (x + ((2*y-1 + 2*L) % (2*L))*L);

                int n_v[4];
                n_v[0] = ((x + 1 + L) % L + y*L);
                n_v[1] = ((x - 1 + L) % L + y*L);
                n_v[2] = (x + (y + 1 + L) % L * L);
                n_v[3] = (x + (y - 1 + L) % L * L);
                int full_edges = 0;

                for (int i = 0; i < 4; i++)
                {
                    if (edges[n_e[i]] < 2)
                    {
                        edges[n_e[i]] += 1;
                        updated = true;
                        if (edges[n_e[i]] == 2)
                        {
                            if (find(v) != find(n_v[i]))
                            {
                                edges[n_e[i]] = 3;
                            }
                            union_func(find(v), n_v[i]);
                            full_edges += 1;
                            if (boundaries.count(cluster) && size[find(n_v[i])] == 1)
                            {
                                boundaries[cluster].push_back(n_v[i]);
                            }
                        }
                    }
                    else
                    {
                        full_edges += 1;
                    }
                }
                if (full_edges >= 4 && boundaries.count(cluster))
                {
                    int index = -1;
                    for (int i = 0; i < boundaries[cluster].size(); i++)
                    {
                        if (boundaries[cluster][i] == v)
                        {
                            index = i;
                            break;
                        }
                    }
                    if (index != -1)
                    {
                        boundaries[cluster].erase(boundaries[cluster].begin() + index);
                    }
                }
            }
        }
    }

    for (int i : remove_cluster)
    {
        clusters.erase(i);
    }
    for (int i : remove_boundary)
    {
        boundaries.erase(i);
    }

    for (auto pair : add_boundary)
    {
        if (!boundaries.count(pair.first))
        {
            boundaries[pair.first] = vector<int>();
        }
        boundaries[pair.first].insert(boundaries[pair.first].end(), pair.second.begin(), pair.second.end());
    }

    return updated;

    /*def iterate(self):
        updated = False
        for cluster in self.clusters:
            if (self.parity[cluster] == 1):
                for v in list(self.boundaries[cluster]):
                    neighbors = self.get_neighbors(v)

                    #experimental
                    full_edges = 0
                    for edge in neighbors:
                        if (self.edges[edge[0]] < 2):
                            self.edges[edge[0]] += 1
                            updated = True
                            if (self.edges[edge[0]] == 2):
                                if (self.find(v) != self.find(edge[1])):
                                    self.edges[edge[0]] = 3
                                self.union(self.find(v), self.find(edge[1]))
                                full_edges += 1
                                if cluster in self.boundaries.keys() and self.size[self.find(edge[1])] == 1:
                                    self.boundaries[cluster].append(edge[1])
                        else:
                            full_edges += 1
                    
                    if (full_edges >= 4 and cluster in self.boundaries and v in self.boundaries[cluster]):
                        self.boundaries[cluster].remove(v)
        if (not updated):
            self.updated = False*/
}

bool ToricSimulator::is_leaf(int i)
{
    int x = i % L;
    int y = int(i / L);
    int n_e[4];
    n_e[0] = (x + 2*y*L);
    n_e[1] = (((x-1 + L) % L) + 2*y*L);
    n_e[2] = (x + ((2*y+1 + 2*L) % (2*L))*L);
    n_e[3] = (x + ((2*y-1 + 2*L) % (2*L))*L);

    int count = 0;
    for (int i = 0; i < 4; i++)
    {
        if (edges[n_e[i]] == 3)
        {
            count++;
        }
    }
    return count == 1;
}

void ToricSimulator::get_leaves()
{
    for (int i = 0; i < L*L; i++)
    {
        if (is_leaf(i))
        {
            //cout<<"There are leaves"<<endl;
            if (leaves.count(find(i)))
            {
                leaves[find(i)].push_back(i);
            }
            else
            {
                leaves[find(i)] = {i};
            }
        }
    }
}

bool ToricSimulator::peel()
{
    //cout<<"Peeling"<<endl;
    get_leaves();
    bool updated = false;
    map<int, vector<int>> remove_leaves;
    map<int, vector<int>> add_leaves;
    for (int cluster : clusters)
    {
        for (int v : leaves[cluster])
        {
            int x = v % L;
            int y = int(v / L);
            int n_e[4];
            n_e[0] = (x + 2*y*L);
            n_e[1] = (((x-1 + L) % L) + 2*y*L);
            n_e[2] = (x + ((2*y+1 + 2*L) % (2*L))*L);
            n_e[3] = (x + ((2*y-1 + 2*L) % (2*L))*L);

            int n_v[4];
            n_v[0] = ((x + 1 + L) % L + y*L);
            n_v[1] = ((x - 1 + L) % L + y*L);
            n_v[2] = (x + (y + 1 + L) % L * L);
            n_v[3] = (x + (y - 1 + L) % L * L);

            for (int i = 0; i < 4; i++)
            {
                if (edges[n_e[i]] == 3)
                {
                    updated = true;
                    if (is_syndrome[v])
                    {
                        is_syndrome[v] ^= 1;
                        is_syndrome[n_v[i]] ^= 1;
                        edges[n_e[i]] = 0;
                        output[n_e[i]] = 1;
                    }
                    else
                    {
                        edges[n_e[i]] = 0;
                    }
                }
                else
                {
                    edges[n_e[i]] = 0;
                }
                if (is_leaf(n_v[i]))
                {
                    bool found = false;
                    for (int j = 0; j < leaves[cluster].size(); j++)
                    {
                        if (leaves[cluster][j] == n_v[i])
                        {
                            found = true;
                            break;
                        }
                    }
                    if (found)
                    {
                        if (add_leaves.count(cluster) == 0)
                        {
                            add_leaves[cluster] = {n_v[i]};
                        }
                        else
                        {
                            add_leaves[cluster].push_back(n_v[i]);
                        }
                    }
                }
            }
            for (int j = 0; j < leaves[cluster].size(); j++)
            {
                if (leaves[cluster][j] == v)
                {
                    if (remove_leaves.count(cluster) == 0)
                    {
                        remove_leaves[cluster] = {v};
                    }
                    else
                    {
                        remove_leaves[cluster].push_back(v);
                    }
                }
            }
        }
    }

    for (int cluster : clusters)
    {
        for (int v : add_leaves[cluster])
        {
            leaves[cluster].push_back(v);
        }
        for (int v : remove_leaves[cluster])
        {
            int index = -1;
            for (int i = 0; i < leaves[cluster].size(); i++)
            {
                if (leaves[cluster][i] == v)
                {
                    index = i;
                    break;
                }
            }
            if (index != -1)
            {
                leaves[cluster].erase(leaves[cluster].begin() + index);
            }
        }
    }

    return updated;

    /*if (not self.got_leaves):
            self.get_leaves()
            self.got_leaves = True
        for cluster in self.clusters:
            for v in list(self.leaves[cluster]):
                #if self.is_leaf(v):
                neighbors = self.get_neighbors(v)

                for n in neighbors:
                    if (self.edges[n[0]] == 3):
                        if (self.is_syndrome[v]):
                            self.is_syndrome[n[1]] ^= 1
                            self.is_syndrome[v] ^= 1
                            self.edges[n[0]] = 0
                            self.tree[n[0]] = 4
                        else:
                            self.edges[n[0]] = 0
                    else:
                        self.edges[n[0]] = 0
                    if self.is_leaf(n[1]) and n[1] not in self.leaves[cluster]:
                        self.leaves[cluster].append(n[1])
                    
                if v in self.leaves[cluster]:
                    self.leaves[cluster].remove(v)*/
}

int main()
{
    srand(time(NULL));
    /*ToricSimulator toric;

    char c;
    while ((cin >> c) && toric.iterate())
    {
        if (c == 'q')
        {
            break;
        }
        toric.display2(toric.parent, toric.edges);
        //cout<<"------------------"<<endl;
        toric.display(toric.parity, toric.qubits);
    }
    toric.get_leaves();

    while ((cin >> c) && toric.peel())
    {
        //cout<<"What's going on here bruh"<<endl;
        if (c == 'q')
        {
            break;
        }
        toric.display2(toric.parent, toric.edges);
        //cout<<"------------------"<<endl;
        //cout<<"Answer:"<<endl;
        toric.display(toric.parity, toric.output);
    }*/

    ostringstream null_stream;
    auto* old_buf = cout.rdbuf(null_stream.rdbuf());

    int iterations = 1000;
    double total_time = 0;

    for (int i = 0; i < iterations; i++)
    {
        ToricSimulator toric;
        auto start_time = chrono::steady_clock::now();
        //cout<<"New toric: \n\n"<<endl;
        //toric.display2(toric.parent, toric.edges);
        //cout<<"------------------"<<endl;
        //toric.display(toric.parity, toric.qubits);
        //cout<<"------------------"<<endl;

        while (toric.iterate());
        toric.get_leaves();
        while (toric.peel());
        
        auto end_time = chrono::steady_clock::now();

			  chrono::duration<double> time_passed = end_time - start_time;
			  total_time += time_passed.count();
        //cout<<"---------------------------"<<endl;
        //toric.display2(toric.parent, toric.edges);
        //cout<<"Answer:"<<endl;
        //toric.display(toric.parity, toric.output);

    }

    cout.rdbuf(old_buf);
    cout<<"Time passed: "<<total_time<<" seconds"<<endl;
    cout<<"Lattice size: "<<L<<"x"<<L<<endl;
    cout<<"Iterations: "<<iterations<<endl;
    cout<<"Millseconds per lattice: "<<total_time/iterations*1000<<endl;
}
