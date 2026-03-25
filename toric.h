#include <iostream>
#include <random>
#include <vector>
#include <map>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <sstream>

using namespace std;

constexpr int L = 64;
constexpr double error_rate = 0.05;

class ToricSimulator
{
    public:
        int qubits[2*L*L];

        //union find structures
        int is_syndrome[2*L*L];
        int parent[2*L*L];
        int size[2*L*L];
        int parity[2*L*L];
        //vector<int> clusters;
        unordered_set<int> clusters;
        map<int, vector<int>> boundaries;
        map<int, vector<int>> leaves;
        int edges[2*L*L];
        int output[2*L*L];

        ToricSimulator();
        void add_error();
        void get_syndromes();
        int find(int i);
        void union_func(int i, int j);
        bool iterate();

        //peeling
        bool is_leaf(int i);
        void get_leaves();
        bool peel();

        //removing things
        unordered_set<int> remove_cluster;
        unordered_set<int> remove_boundary;
        map<int, vector<int>> add_boundary;
        bool updated = false;

        //display
        void display(int* vertices, int* edges);
        void display2(int* vertices, int* edges);
        
        /*self.support = [[0 for _ in range(L)] for _ in range (L)]
        self.qubits = [[0 for _ in range(L)] for _ in range (L*2)]

        #union find things
        self.is_syndrome = [0 for _ in range(2*L*L)]
        self.display_syndrome = []
        self.parent = [x for x in range(2*L*L)]
        self.size = [1 for x in range(2*L*L)]
        self.parity = [0 for x in range(2*L*L)]
        self.clusters = []
        self.boundaries = {}
        self.edges = [0 for x in range(2*L*L)]
        self.tree = [0 for x in range(2*L*L)]
        self.leaves = {}
        self.got_leaves = False
        self.updated = True*/
};