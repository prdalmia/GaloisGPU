/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"
inline __device__ void cudaBarrierAtomicSubSRB(unsigned int * globalBarr,
  // numBarr represents the number of
  // TBs going to the barrier
  const unsigned int numBarr,
  int backoff,
  const bool isMasterThread,
  bool * volatile sense,
  bool * volatile global_sense)
{
__syncthreads();
if (isMasterThread)
{
//printf("Inside global Barrier for blockID %d and sense is %d and global sense is %d\n", blockIdx.x, *sense, *global_sense);
// atomicInc acts as a store release, need TF to enforce ordering
__threadfence();
// atomicInc effectively adds 1 to atomic for each TB that's part of the
// global barrier.
atomicInc(globalBarr, 0x7FFFFFFF);
//printf("Global barr is %d\n", *globalBarr);
}
__syncthreads();

while (*global_sense != *sense)
{
if (isMasterThread)
{
//printf("Global sense hili\n");
/*
For the tree barrier we expect only 1 TB from each SM to enter the
global barrier.  Since we are assuming an equal amount of work for all
SMs, we can use the # of TBs reaching the barrier for the compare value
here.  Once the atomic's value == numBarr, then reset the value to 0 and
proceed because all of the TBs have reached the global barrier.
*/
if (atomicCAS(globalBarr, numBarr, 0) == numBarr) {
// atomicCAS acts as a load acquire, need TF to enforce ordering
__threadfence();
*global_sense = *sense;
}
else { // increase backoff to avoid repeatedly hammering global barrier
// (capped) exponential backoff
backoff = (((backoff << 1) + 1) & (1024-1));
}
}
__syncthreads();

// do exponential backoff to reduce the number of times we pound the global
// barrier
if(isMasterThread){
//if (*global_sense != *sense) {
for (int i = 0; i < backoff; ++i) { ; }
}
__syncthreads();
//}
}
}

inline __device__ void cudaBarrierAtomicSRB(unsigned int * barrierBuffers,
// numBarr represents the number of
// TBs going to the barrier
const unsigned int numBarr,
const bool isMasterThread,
bool * volatile sense,
bool * volatile global_sense)
{
__shared__ int backoff;

if (isMasterThread) {
backoff = 1;
}
__syncthreads();

cudaBarrierAtomicSubSRB(barrierBuffers, numBarr, backoff, isMasterThread, sense, global_sense);
}

inline __device__ void cudaBarrierAtomicSubLocalSRB(unsigned int * perSMBarr,
       const unsigned int numTBs_thisSM,
       const bool isMasterThread,
       bool * sense,
       const int smID,
       unsigned int* last_block)

{
__syncthreads();
__shared__ bool s;
if (isMasterThread)
{
s = !(*sense);
// atomicInc acts as a store release, need TF to enforce ordering locally
__threadfence_block();
/*
atomicInc effectively adds 1 to atomic for each TB that's part of the
barrier.  For the local barrier, this requires using the per-CU
locations.
*/
atomicInc(perSMBarr, 0x7FFFFFFF);
}
__syncthreads();

while (*sense != s)
{
if (isMasterThread)
{
/*
Once all of the TBs on this SM have incremented the value at atomic,
then the value (for the local barrier) should be equal to the # of TBs
on this SM.  Once that is true, then we want to reset the atomic to 0
and proceed because all of the TBs on this SM have reached the local
barrier.
*/
if (atomicCAS(perSMBarr, numTBs_thisSM, 0) == numTBs_thisSM) {
// atomicCAS acts as a load acquire, need TF to enforce ordering
// locally
__threadfence_block();
*sense = s;
*last_block = blockIdx.x;
}
}
__syncthreads();
}
}

//Implements PerSM sense reversing barrier
inline __device__ void cudaBarrierAtomicLocalSRB(unsigned int * perSMBarrierBuffers,
             unsigned int * last_block,
             const unsigned int smID,
             const unsigned int numTBs_thisSM,
             const bool isMasterThread,
             bool* sense)
{
// each SM has MAX_BLOCKS locations in barrierBuffers, so my SM's locations
// start at barrierBuffers[smID*MAX_BLOCKS]
cudaBarrierAtomicSubLocalSRB(perSMBarrierBuffers, numTBs_thisSM, isMasterThread, sense, smID, last_block);
}

/*
Helper function for joining the barrier with the atomic tree barrier.
*/
__device__ void joinBarrier_helperSRB(bool * global_sense,
bool * perSMsense,
bool * done,
unsigned int* global_count,
unsigned int* local_count,
unsigned int* last_block,
const unsigned int numBlocksAtBarr,
const int smID,
const int perSM_blockID,
const int numTBs_perSM,
const bool isMasterThread) {                                 
__syncthreads();
if (numTBs_perSM > 1) {
cudaBarrierAtomicLocalSRB(&local_count[smID], &last_block[smID], smID, numTBs_perSM, isMasterThread, &perSMsense[smID]);

// only 1 TB per SM needs to do the global barrier since we synchronized
// the TBs locally first
if (blockIdx.x == last_block[smID]) {
  if(isMasterThread && perSM_blockID == 0){    
  }
  __syncthreads();
cudaBarrierAtomicSRB(global_count, numBlocksAtBarr, isMasterThread , &perSMsense[smID], global_sense);  
//*done = 1;
}
else {
if(isMasterThread){
while (*global_sense != perSMsense[smID] ){  
__threadfence();
}
}
__syncthreads();
}    
} else { // if only 1 TB on the SM, no need for the local barriers
cudaBarrierAtomicSRB(global_count, numBlocksAtBarr, isMasterThread,  &perSMsense[smID], global_sense);
}
}


__device__ void kernelAtomicTreeBarrierUniqSRB( bool * global_sense,
bool * perSMsense,
bool * done,
unsigned int* global_count,
unsigned int* local_count,
unsigned int* last_block,
const int NUM_SM)
{

// local variables
// thread 0 is master thread
const bool isMasterThread = ((threadIdx.x == 0) && (threadIdx.y == 0) &&
(threadIdx.z == 0));
// represents the number of TBs going to the barrier (max NUM_SM, gridDim.x if
// fewer TBs than SMs).
const unsigned int numBlocksAtBarr = ((gridDim.x < NUM_SM) ? gridDim.x :
NUM_SM);
const int smID = (blockIdx.x % numBlocksAtBarr); // mod by # SMs to get SM ID

// all thread blocks on the same SM access unique locations because the
// barrier can't ensure DRF between TBs
const int perSM_blockID = (blockIdx.x / numBlocksAtBarr);
// given the gridDim.x, we can figure out how many TBs are on our SM -- assume
// all SMs have an identical number of TBs

int numTBs_perSM = (int)ceil((float)gridDim.x / numBlocksAtBarr);

joinBarrier_helperSRB(global_sense, perSMsense, done, global_count, local_count, last_block,
numBlocksAtBarr, smID, perSM_blockID, numTBs_perSM,
isMasterThread);
}

void kernel_sizing(CSRGraphTex &, dim3 &, dim3 &);
#define TB_SIZE 32
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=False $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=1 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=False $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=texture $ cuda.use_worklist_slots=True $ cuda.worklist_type=texture";
AppendOnlyList el;
#include "mst.h"
#define INF UINT_MAX
const int DEBUG = 0;
static const int __tb_union_components = TB_SIZE;
__global__ void init_wl(CSRGraphTex graph, WorklistT in_wl, WorklistT out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type node_end;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    (out_wl).push(node);
  }
}
__global__ void find_comp_min_elem(CSRGraphTex graph, struct comp_data comp, LockArrayTicket complocks, ComponentSpace cs, int level, WorklistT in_wl, WorklistT out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlnode_end;

  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    index_type edge_end;
    pop = (in_wl).pop_id(wlnode, node);
    unsigned minwt = INF;
    unsigned minedge = INF;
    int degree = graph.getOutDegree(node);
    int mindstcomp  = 0;
    int srccomp = cs.find(node);
    edge_end = (graph).getFirstEdge((node) + 1);
    for (index_type edge = (graph).getFirstEdge(node) + 0; edge < edge_end; edge += 1)
    {
      int edgewt = graph.getAbsWeight(edge);
      if (edgewt < minwt)
      {
        int dstcomp = cs.find(graph.getAbsDestination(edge));
        if (dstcomp != srccomp)
        {
          minwt = edgewt;
          minedge = edge;
        }
      }
    }
    if (minwt != INF)
    {
      (out_wl).push(node);
      {
        volatile bool done_ = false;
		int _ticket = (complocks).reserve(srccomp);
        while (!done_)
        {
          if (complocks.acquire_or_fail(srccomp, _ticket))
          {
            if (comp.minwt[srccomp] == 0 || (comp.lvl[srccomp] < level) || (comp.minwt[srccomp] > minwt))
            {
              comp.minwt[srccomp] = minwt;
              comp.lvl[srccomp] = level;
              comp.minedge[srccomp] = minedge;
            }
            complocks.release(srccomp);
            done_ = true;
          }
        }
      }
    }
    else
    {
      if (cs.isBoss(node) && degree)
      {
        (out_wl).push(node);
      }
    }
  }
}
__global__ void union_components(CSRGraphTex graph, ComponentSpace cs, struct comp_data compdata, int level, AppendOnlyList el, AppendOnlyList ew, WorklistT in_wl, WorklistT out_wl, GlobalBarrier gb, Any ret_val, bool * global_sense,
  bool * perSMsense,
  bool * done,
  unsigned int* global_count,
  unsigned int* local_count,
  unsigned int* last_block,
  const int NUM_SM)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlnode_end;
  wlnode_end = roundup((*((volatile index_type *) (in_wl).dindex)), (nthreads));
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    pop = (in_wl).pop_id(wlnode, node);
    int r = 0;
    int dstcomp = -1;
    int srccomp = -1;
    if (pop && compdata.lvl[node] == level)
    {
      srccomp = cs.find(node);
      dstcomp = cs.find(graph.getAbsDestination(compdata.minedge[node]));
    }
    //gb.Sync();
    kernelAtomicTreeBarrierUniqSRB(global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM);     
    if (srccomp != dstcomp)
    {
      if (!cs.unify(srccomp, dstcomp))
      {
        r = 1;
      }
      else
      {
        el.push(compdata.minedge[node]);
        ew.push(compdata.minwt[node]);
      }
    }
    //gb.Sync();
    kernelAtomicTreeBarrierUniqSRB(global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM);     
    if (r)
    {
      ret_val.return_(true);
      continue;
    }
  }
}
void gg_main(CSRGraphTex& hg, CSRGraphTex& gg)
{
  dim3 blocks, threads;
  kernel_sizing(gg, blocks, threads);
  blocks = ggc_get_nSM()*32;
  static GlobalBarrierLifetime union_components_barrier;
  static bool union_components_barrier_inited;
  struct comp_data comp;
  PipeContextT<WorklistT> pipe;
  ComponentSpace cs (hg.nnodes);
  el = AppendOnlyList(hg.nedges);
  AppendOnlyList ew (hg.nedges);
  bool * global_sense;
    bool* perSMsense;
    bool * done;
    unsigned int* global_count;
    unsigned int* local_count; 
    unsigned int *last_block;
    int NUM_SM = ggc_get_nSM();
    cudaMallocManaged((void **)&global_sense,sizeof(bool));
    cudaMallocManaged((void **)&done,sizeof(bool));
    cudaMallocManaged((void **)&perSMsense,NUM_SM*sizeof(bool));
    cudaMallocManaged((void **)&last_block,sizeof(unsigned int)*(NUM_SM));
    cudaMallocManaged((void **)&local_count,  NUM_SM*sizeof(unsigned int));
    cudaMallocManaged((void **)&global_count,sizeof(unsigned int));
    
    cudaMemset(global_sense, false, sizeof(bool));
    cudaMemset(done, false, sizeof(bool));
    cudaMemset(global_count, 0, sizeof(unsigned int));

    for (int i = 0; i < NUM_SM; ++i) {
       cudaMemset(&perSMsense[i], false, sizeof(bool));
       cudaMemset(&local_count[i], 0, sizeof(unsigned int));
       cudaMemset(&last_block[i], 0, sizeof(unsigned int));
     }
  static const size_t union_components_residency = maximum_residency(union_components, __tb_union_components, 0);
  static const size_t union_components_blocks = GG_MIN(blocks.x, ggc_get_nSM() * union_components_residency);
  if(!union_components_barrier_inited) { union_components_barrier.Setup(union_components_blocks); union_components_barrier_inited = true;};
  comp.weight.alloc(hg.nnodes);
  comp.edge.alloc(hg.nnodes);
  comp.node.alloc(hg.nnodes);
  comp.level.alloc(hg.nnodes);
  comp.dstcomp.alloc(hg.nnodes);
  comp.lvl = comp.level.zero_gpu();
  comp.minwt = comp.weight.zero_gpu();
  comp.minedge = comp.edge.gpu_wr_ptr();
  comp.minnode = comp.node.gpu_wr_ptr();
  comp.mindstcomp = comp.dstcomp.gpu_wr_ptr();
  LockArrayTicket complocks (hg.nnodes);
  int level = 1;
  int mw = 0;
  int last_mw = 0;
  pipe = PipeContextT<WorklistT>(hg.nnodes);
  {
    {
      pipe.out_wl().will_write();
      init_wl <<<blocks, threads>>>(gg, pipe.in_wl(), pipe.out_wl());
      pipe.in_wl().swap_slots();
      pipe.advance2();
      while (pipe.in_wl().nitems())
      {
        bool loopc = false;
        last_mw = mw;
        pipe.out_wl().will_write();
        find_comp_min_elem <<<blocks, threads>>>(gg, comp, complocks, cs, level, pipe.in_wl(), pipe.out_wl());
        pipe.in_wl().swap_slots();
        pipe.advance2();
        do
        {
          Shared<int> retval = Shared<int>(1);
          Any _rv;
          *(retval.cpu_wr_ptr()) = 0;
          _rv.rv = retval.gpu_wr_ptr();
          pipe.out_wl().will_write();
          union_components <<<union_components_blocks, __tb_union_components>>>(gg, cs, comp, level, el, ew, pipe.in_wl(), pipe.out_wl(), union_components_barrier, _rv, global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM);
          loopc = *(retval.cpu_rd_ptr()) > 0;
        }
        while (loopc);
        mw = el.nitems();
        level++;
        if (last_mw == mw)
        {
          break;
        }
      }
    }
  }
  unsigned long int rweight = 0;
  size_t nmstedges ;
  nmstedges = ew.nitems();
  mgpu::Reduce(ew.list.gpu_rd_ptr(), nmstedges, (long unsigned int)0, mgpu::plus<long unsigned int>(), (long unsigned int*)0, &rweight, *mgc);
  printf("number of iterations: %d\n", level);
  printf("final mstwt: %llu\n", rweight);
  printf("total edges: %llu, total components: %llu\n", nmstedges, cs.numberOfComponentsHost());
  cudaFree(global_sense);
  cudaFree(perSMsense);
  cudaFree(last_block);
  cudaFree(local_count);
  cudaFree(global_count);
  cudaFree(done);
}
