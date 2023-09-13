import hybrid
import dimod
from dimod import BinaryQuadraticModel,quicksum,Binary
from dwave.system import LeapHybridBQMSampler
import re
from dwave.system.samplers import DWaveSampler
from hybrid import traits
from dwave.system.composites import AutoEmbeddingComposite, FixedEmbeddingComposite
from hybrid.core import Runnable, SampleSet
from dwave.preprocessing.composites import SpinReversalTransformComposite

class QPUSubproblemExternalEmbeddingSampler(traits.SubproblemSampler,
                                            traits.EmbeddingIntaking,
                                            traits.SISO, Runnable):
    r"""A quantum sampler for a subproblem with a defined minor-embedding.
    Note:
        Externally supplied embedding must be present in the input state.
    Args:
        num_reads (int, optional, default=100):
            Number of states (output solutions) to read from the sampler.
        qpu_sampler (:class:`dimod.Sampler`, optional, default=\ :class:`~dwave.system.samplers.DWaveSampler()` ):
            Quantum sampler such as a D-Wave system.
        sampling_params (dict):
            Dictionary of keyword arguments with values that will be used
            on every call of the (external-embedding-wrapped QPU) sampler.
        logical_srt (int, optional, default=False):
            Perform a spin-reversal transform over the logical space.
    See :ref:`samplers-examples`.
    """

    def __init__(self, num_reads=100, qpu_sampler=None, sampling_params=None,
                 logical_srt=False, **runopts):
        super(QPUSubproblemExternalEmbeddingSampler, self).__init__(**runopts)

        self.num_reads = num_reads

        if qpu_sampler is None:
            qpu_sampler = DWaveSampler()
        self.sampler = qpu_sampler

        if sampling_params is None:
            sampling_params = {}
        self.sampling_params = sampling_params

        self.logical_srt = logical_srt
        self.qpu_access_time=0
    def __repr__(self):
        return ("{self}(num_reads={self.num_reads!r}, "
                       "qpu_sampler={self.sampler!r}, "
                       "sampling_params={self.sampling_params!r})").format(self=self)

    def next(self, state, **runopts):
        num_reads = runopts.get('num_reads', self.num_reads)
        sampling_params = runopts.get('sampling_params', self.sampling_params)

        params = sampling_params.copy()
        params.update(num_reads=num_reads)

        sampler = FixedEmbeddingComposite(self.sampler, embedding=state.embedding)
        if self.logical_srt:
            params.update(num_spin_reversal_transforms=1)
            sampler = SpinReversalTransformComposite(sampler)
        response = sampler.sample(state.subproblem, **params)
        self.qpu_access_time+=response.info['timing']['qpu_access_time']*(0.001)
        return state.updated(subsamples=response)


class KerberosSampler(dimod.Sampler):

    """An opinionated dimod-compatible hybrid asynchronous decomposition sampler
    for problems of arbitrary structure and size.

    Examples:
        This example solves a two-variable Ising model.

        >>> import dimod
        >>> import hybrid
        >>> response = hybrid.KerberosSampler().sample_ising(
        ...                     {'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})    # doctest: +SKIP
        >>> response.data_vectors['energy']      # doctest: +SKIP
        array([-1.5])

    """
    properties = None
    parameters = None
    runnable = None

    def __init__(self):
        self.parameters = {
            'num_reads': [],
            'init_sample': [],
            'max_iter': [],
            'max_time': [],
            'convergence': [],
            'energy_threshold': [],
            'sa_reads': [],
            'sa_sweeps': [],
            'tabu_timeout': [],
            'qpu_reads': [],
            'qpu_sampler': [],
            'qpu_params': [],
            'max_subproblem_size': []
        }
        self.properties = {}
        self.QPUSubproblemExternalEmbeddingSampler=QPUSubproblemExternalEmbeddingSampler
    def Kerberos(self,sampler,max_iter=100, max_time=None, convergence=3, energy_threshold=None,
                 sa_reads=1, sa_sweeps=10000, tabu_timeout=500,
                 qpu_reads=100, qpu_sampler=None, qpu_params=None,
                 max_subproblem_size=50):
        """An opinionated hybrid asynchronous decomposition sampler for problems of
        arbitrary structure and size. Runs Tabu search, Simulated annealing and QPU
        subproblem sampling (for high energy impact problem variables) in parallel
        and returns the best samples.

        Kerberos workflow is used by :class:`KerberosSampler`.


        Returns:
            Workflow (:class:`~hybrid.core.Runnable` instance).

        """
        #=self.QPUSubproblemAutoEmbeddingSampler(num_reads=qpu_reads, qpu_sampler=qpu_sampler, sampling_params=qpu_params)
        energy_reached = None
        if energy_threshold is not None:
            energy_reached = lambda en: en <= energy_threshold
        iteration = hybrid.Race(
            hybrid.BlockingIdentity(),
            hybrid.InterruptableTabuSampler(
                timeout=tabu_timeout),
            hybrid.InterruptableSimulatedAnnealingProblemSampler(
                num_reads=sa_reads, num_sweeps=sa_sweeps),
            hybrid.EnergyImpactDecomposer(
                size=max_subproblem_size, rolling=True, rolling_history=0.3, traversal='bfs')
                | sampler
                | hybrid.SplatComposer()
        ) | hybrid.ArgMin()

        workflow = hybrid.Loop(iteration, max_iter=max_iter, max_time=max_time,
                               convergence=convergence, terminate=energy_reached)
        return workflow

    def sample(self, bqm,init_sample=None, num_reads=1, max_iter=100, max_time=None, convergence=3, energy_threshold=None,
                 sa_reads=1, sa_sweeps=10000, tabu_timeout=500,
                 qpu_reads=100, qpu_sampler=None, qpu_params=None,
                 max_subproblem_size=50):
            """Run Tabu search, Simulated annealing and QPU subproblem sampling (for
            high energy impact problem variables) in parallel and return the best
            samples.

            Sampling Args:

                bqm (:obj:`~dimod.BinaryQuadraticModel`):
                    Binary quadratic model to be sampled from.

                init_sample (:class:`~dimod.SampleSet`, callable, ``None``):
                    Initial sample set (or sample generator) used for each "read".
                    Use a random sample for each read by default.

                num_reads (int):
                    Number of reads. Each sample is the result of a single run of the
                    hybrid algorithm.

            Termination Criteria Args:

                max_iter (int):
                    Number of iterations in the hybrid algorithm.

                max_time (float/None, optional, default=None):
                    Wall clock runtime termination criterion. Unlimited by default.

                convergence (int):
                    Number of iterations with no improvement that terminates sampling.

                energy_threshold (float, optional):
                    Terminate when this energy threshold is surpassed. Check is
                    performed at the end of each iteration.

            Simulated Annealing Parameters:

                sa_reads (int):
                    Number of reads in the simulated annealing branch.

                sa_sweeps (int):
                    Number of sweeps in the simulated annealing branch.

            Tabu Search Parameters:

                tabu_timeout (int):
                    Timeout for non-interruptable operation of tabu search (time in
                    milliseconds).

            QPU Sampling Parameters:

                qpu_reads (int):
                    Number of reads in the QPU branch.

                qpu_sampler (:class:`dimod.Sampler`, optional, default=DWaveSampler()):
                    Quantum sampler such as a D-Wave system.

                qpu_params (dict):
                    Dictionary of keyword arguments with values that will be used
                    on every call of the QPU sampler.

                max_subproblem_size (int):
                    Maximum size of the subproblem selected in the QPU branch.

            Returns:
                :obj:`~dimod.SampleSet`: A `dimod` :obj:`.~dimod.SampleSet` object.

            """
            if callable(init_sample):
                init_state_gen = lambda: hybrid.State.from_sample(init_sample(), bqm)
            elif init_sample is None:
                init_state_gen = lambda: hybrid.State.from_sample(hybrid.random_sample(bqm), bqm)
            elif isinstance(init_sample, dimod.SampleSet):
                init_state_gen = lambda: hybrid.State.from_sample(init_sample, bqm)
            else:
                raise TypeError("'init_sample' should be a SampleSet or a SampleSet generator")
            external_sampler=self.QPUSubproblemExternalEmbeddingSampler(num_reads=qpu_reads, qpu_sampler=qpu_sampler, sampling_params=qpu_params)
            sampler=hybrid.SubproblemCliqueEmbedder(sampler=qpu_sampler,) | external_sampler
            #self.QPUSubproblemAutoEmbeddingSampler(num_reads=qpu_reads, qpu_sampler=qpu_sampler, sampling_params=qpu_params)
            self.runnable = self.Kerberos(sampler,max_iter, max_time, convergence, energy_threshold,sa_reads, sa_sweeps, tabu_timeout,qpu_reads, qpu_sampler, qpu_params,
                         max_subproblem_size)

            samples = []
            energies = []
            for _ in range(num_reads):
                init_state = init_state_gen()
                final_state = self.runnable.run(init_state)
                # the best sample from each run is one "read"
                ss = final_state.result().samples
                ss.change_vartype(bqm.vartype, inplace=True)
                samples.append(ss.first.sample)
                energies.append(ss.first.energy)

            return dimod.SampleSet.from_samples(samples, vartype=bqm.vartype, energy=energies),external_sampler.qpu_access_time



def call_bqm_solver(self,max_iter,convergence,num_reads,solver):
    """Calls bqm solver.
    Args:
        time_limit: time limit in second
        bqm: constrained quadratic model
    """
    sampler,qpu_access_time = self.KerberosSampler.sample(self.bqm,max_iter=max_iter,convergence=convergence,qpu_sampler=solver,qpu_params={'label': 'JSSP_bqm_iter'},qpu_reads=num_reads)
    samplerSET=sampler.samples()
    solution=[]
    for key in samplerSET:
        sampler =key
    for key in sampler:
        if sampler[key] !=0 and 'slack' not in key:
            solution.append((key,sampler[key]))
    return solution,qpu_access_time

def call_bqm_solver_classic(model:BinaryQuadraticModel,convergence,sa_num_reads=1,sa_num_sweeps=10000,subproblem_size=2):#write the related parameters
    """classic solver use the simulated annealing and tabu search parallel and get the best result"""
    sa_num_reads=50
    sa_num_sweeps=10000
    subproblem_size=2
    """iteration = hybrid.RacingBranches(
    hybrid.InterruptableTabuSampler(num_reads=50, tenure=None, timeout=100, initial_states_generator='random'),
    hybrid.EnergyImpactDecomposer(size=subproblem_size)
    | hybrid.SimulatedAnnealingSubproblemSampler(num_reads=sa_num_reads, num_sweeps=sa_num_sweeps,
                beta_range=None, beta_schedule_type='geometric',
                initial_states_generator='random')
    | hybrid.SplatComposer()
    ) | hybrid.ArgMin()
    workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=convergence)
    # Solve the problem
    init_state = hybrid.State.from_problem(self.bqm)
    final_state = workflow.run(init_state).result()
    solution=[]
    for key in final_state.samples.first.sample:   #get the solution
        if final_state.samples.first.sample[key] !=0:
            solution.append((key,final_state.samples.first.sample[key]))
    qpu_access_time=0"""
    workflow = hybrid.Parallel(
        hybrid.SimulatedAnnealingProblemSampler(num_reads=50),
        hybrid.TabuProblemSampler(num_reads=50)) | hybrid.MergeSamples()
    init_state = hybrid.State.from_problem(model)
    final_state = workflow.run(init_state).result()

    solution = final_state.samples.first.sample
    qpu_access_time=0
    return solution, qpu_access_time

class resilience_computation():
    def __init__(self):
        self.max_suppliers=6
        self.min_suppliers=2
        self.max_warehouses_stock=6
        self.max_manufacturing_units=6
        self.min_manufacturing_units=3
        self.max_distribution_unit_usage=4
        self.min_distribution_unit_usage=1
        self.cost_for_supplier=10000
        self.cost_for_canceling_supply=5000
        self.cost_for_manufacturing_unit=15000
        self.cost_for_warehouses=8000
        self.cost_production_of_material=5000
        self.revenue_for_product=100000
        self.cost_for_unmet_customer_demand=17000

        self.X_jd={}
        self.X_ns={}
        self.X_km={}
        self.X_yrm={}
        self.X_vp={}
        self.X_hm={}
        self.KerberosSampler=KerberosSampler()
        """
        #Initial State:
        self.initial_suplliers=4
        self.initial_manufacturing_units=4
        self.initial_used_manufacturing_units=4
        self.initial_distribution_centercapacity=4
        self.initial_warehouse_stock_raw_material=0
        self.initial_warehouse_stock_product=0
        self.initial_customer_demand=4
        self.initial_warehouse_stock_raw_material1=0
        self.initial_warehouse_stock_product1=0
        """

        """#S1:canceled customer lead to one product in the warehouse and three products for the distribution center
        self.initial_suplliers=4
        self.initial_manufacturing_units=4
        self.initial_used_manufacturing_units=4
        self.initial_distribution_centercapacity=3
        self.initial_warehouse_stock_raw_material=0
        self.initial_warehouse_stock_product=0
        self.initial_customer_demand=3
        self.initial_warehouse_stock_raw_material1=0
        self.initial_warehouse_stock_product1=1"""



        """#S2:The distribution center can only send 3 products to customers. But the amount of customers are still 4. It leads to the cost for unmeted customer

        self.initial_suplliers=4
        self.initial_manufacturing_units=4
        self.initial_used_manufacturing_units=4
        self.initial_distribution_centercapacity=3
        self.initial_warehouse_stock_raw_material=0
        self.initial_warehouse_stock_product=0
        self.initial_customer_demand=4
        self.initial_warehouse_stock_raw_material1=0
        self.initial_warehouse_stock_product1=1"""



        #S3:One manufacturing unit is broken down. It leads that the difference between the used manufacturing units and total manufacturing units is one.
        self.initial_suplliers=4
        self.initial_manufacturing_units=4
        self.initial_used_manufacturing_units=3
        self.initial_distribution_centercapacity=3
        self.initial_warehouse_stock_raw_material=0
        self.initial_warehouse_stock_product=0
        self.initial_customer_demand=4
        self.initial_warehouse_stock_raw_material1=1
        self.initial_warehouse_stock_product1=0

    def calculate_initial_profit(self):#calculate the profit after the disturbance without the strategy
        self.revenue=self.revenue_for_product*self.initial_distribution_centercapacity
        self.profit=(self.revenue-self.initial_manufacturing_units*self.cost_for_manufacturing_unit-self.initial_suplliers*self.cost_for_supplier
                -(self.initial_warehouse_stock_product1+self.initial_warehouse_stock_raw_material1)*self.cost_for_warehouses
            -self.initial_used_manufacturing_units*self.cost_production_of_material-self.cost_for_unmet_customer_demand*(self.initial_customer_demand-self.initial_distribution_centercapacity))

        print(self.profit)
        return self.profit



    def define_variables(self):#define the variables that we want to find out (in this case:the strategy,i.e., the number of elements in the supply chain )
        self.X_jd={(i): 'x_d_{}'.format(i)
              for i in range (self.min_distribution_unit_usage,self.initial_customer_demand+1)}

        self.X_ns={(i): 'x_s_{}'.format(i)
              for i in range (self.min_suppliers,self.max_suppliers+1)}

        self.X_km={(i): 'x_mu_{}'.format(i)#used machines
              for i in range (self.min_manufacturing_units,self.max_manufacturing_units+1)}

        self.X_hm={(i): 'x_m_{}'.format(i)
              for i in range (self.initial_manufacturing_units,self.max_manufacturing_units+1)}

        self.X_yrm={(i): 'x_rm_{}'.format(i)
              for i in range (self.max_warehouses_stock+1)}

        self.X_vp={(i): 'x_p_{}'.format(i)
              for i in range (self.max_warehouses_stock+1)}

        self.variables=[self.X_jd,self.X_ns,self.X_km,self.X_yrm,self.X_vp,self.X_hm]
        return self.variables


    def define_bqm(self):
        """define bqm model
        For the bqm model, the variables should be added to the bqm model by the command "bqm.add_variable" """
        self.bqm=BinaryQuadraticModel('BINARY')
        for i in self.variables:
            for j in i.values():
                self.bqm.add_variable(j)
        return self.bqm


    def constraint_used_machines(self,weight):

        mu=[(self.X_km[(i)],i) for i in range(self.min_manufacturing_units,self.max_manufacturing_units+1)]#the amount of used machine
        mu_negativ=[(self.X_km[(i)],-i) for i in range(self.min_manufacturing_units,self.max_manufacturing_units+1)]
        m=[(self.X_hm[(i)],i) for i in range(self.initial_manufacturing_units,self.max_manufacturing_units+1)]#the number of total machine
        """
        #S1,S2 m-mu>=0
        self.bqm.add_linear_inequality_constraint(m+mu_negativ, lagrange_multiplier=weight,lb=0,ub=self.max_manufacturing_units, label="used_machines")  #create the inequality equation
        """
        #S3 m-mu>=1
        self.bqm.add_linear_inequality_constraint(m+mu_negativ, lagrange_multiplier=weight,lb=1,ub=self.max_manufacturing_units, label="used_machines")

        return

    def constraint_warehouse(self,weight):
        rm=[(self.X_yrm[(i)],i) for i in range(self.max_warehouses_stock+1)]#the number of raw materials in the warehouse
        p=[(self.X_vp[(i)],i) for i in range(self.max_warehouses_stock+1)]#the number of productions in the warehouse
        self.bqm.add_linear_inequality_constraint(rm+p,lagrange_multiplier=weight,lb=0,ub=self.max_warehouses_stock,label="constraint_max_warehousestock")#create the inequality equation  rm+p<=c_w

        return

    def constraint_distribution(self,weight):
        d=[(self.X_jd[(i)],i) for i in range(self.min_distribution_unit_usage,self.initial_customer_demand+1)]

        self.bqm.add_linear_inequality_constraint(d, lagrange_multiplier=weight,lb=0,ub=self.initial_customer_demand, label="constraint_distribution")#create the inequality equation d<=o



        return

    def constraint_transfer_production(self,weight):
        p=[(self.X_vp[(i)],i) for i in range(self.max_warehouses_stock+1)]
        mu=[(self.X_km[(i)],i) for i in range(self.min_manufacturing_units,self.max_manufacturing_units+1)]
        mu_negativ=[(self.X_km[(i)],-i) for i in range(self.min_manufacturing_units,self.max_manufacturing_units+1)]

        d=[(self.X_jd[(i)],i) for i in range(self.min_distribution_unit_usage,self.initial_customer_demand+1)]
        self.bqm.add_linear_equality_constraint(p+mu_negativ+d, lagrange_multiplier=weight, constant=self.initial_warehouse_stock_product)#create linear equlity equation p-mu+d+I_wp=0 (I_wp is the constant)

        return
    def constraint_transfer_raw_material(self,weight):
        rm=[(self.X_yrm[(i)],i) for i in range(self.max_warehouses_stock+1)]
        rm_negativ=[(self.X_yrm[(i)],-i) for i in range(self.max_warehouses_stock+1)]
        mu=[(self.X_km[(i)],i) for i in range(self.min_manufacturing_units,self.max_manufacturing_units+1)]
        mu_negativ=[(self.X_km[(i)],-i) for i in range(self.min_manufacturing_units,self.max_manufacturing_units+1)]
        s_negativ=[(self.X_ns[(i)],-i) for i in range(self.min_suppliers,self.max_suppliers+1)]
        s=[(self.X_ns[(i)],i) for i in range(self.min_suppliers,self.max_suppliers+1)]
        self.bqm.add_linear_equality_constraint(rm_negativ+s+mu_negativ, lagrange_multiplier=weight, constant=self.initial_warehouse_stock_raw_material)#create linear equlity equation s-mu-rm+I_wrm=0
        return

    def constraint_variables(self,weight):# the constraint for the variables. Each element in the supply chain has only one amount.

        self.bqm.add_linear_equality_constraint([(self.X_yrm[(i)],1) for i in range(self.max_warehouses_stock+1)], lagrange_multiplier=weight, constant=-1)
        self.bqm.add_linear_equality_constraint([(self.X_vp[(i)],1) for i in range(self.max_warehouses_stock+1)], lagrange_multiplier=weight, constant=-1)
        self.bqm.add_linear_equality_constraint([(self.X_km[(i)],1) for i in range(self.min_manufacturing_units,self.max_manufacturing_units+1)], lagrange_multiplier=weight, constant=-1)
        self.bqm.add_linear_equality_constraint([(self.X_jd[(i)],1) for i in range(self.min_distribution_unit_usage,self.initial_customer_demand+1)], lagrange_multiplier=weight, constant=-1)
        self.bqm.add_linear_equality_constraint([(self.X_ns[(i)],1) for i in range(self.min_suppliers,self.max_suppliers+1)], lagrange_multiplier=weight, constant=-1)
        self.bqm.add_linear_equality_constraint([(self.X_hm[(i)],1) for i in range(self.initial_manufacturing_units,self.max_manufacturing_units+1)], lagrange_multiplier=weight, constant=-1)
        """#S2: for this disturbance: the distribution center can only send 3 productions. So the binary variables for the distribution centers' number of 3 is 1
        self.bqm.fix_variable(self.X_jd[3], 1)"""
        return
    def objective_cost(self,weight):#penalize the large cost caused by the selected strategy.
        bias={}
        for rm in self.X_yrm:
            bias[self.X_yrm[rm]]=self.cost_for_warehouses*rm*weight

        for p in self.X_vp:
             bias[self.X_vp[p]]=self.cost_for_warehouses*p*weight

        for s in self.X_ns:
            if s <= self.initial_suplliers:#if the supply is canceled, the canceled supply cost should be considered.
                bias[self.X_ns[s]]=(self.cost_for_supplier*s+self.cost_for_canceling_supply*(self.initial_suplliers-s))*weight
            else:
                bias[self.X_ns[s]]=self.cost_for_supplier*s*weight
        for o in self.X_jd:
            bias[self.X_jd[o]]=(self.cost_for_unmet_customer_demand*(self.initial_customer_demand-o)-self.revenue_for_product*o)*weight

        for m in self.X_hm:
            bias[self.X_hm[m]]=self.cost_for_manufacturing_unit*m*weight

        for mu in self.X_km:
            bias[self.X_km[mu]]=self.cost_production_of_material*mu*weight

        self.bqm.add_linear_from(bias)
        return


    def call_bqm_solver(self,max_iter,convergence,num_reads,solver):
        """Calls bqm solver.
        Args:
            time_limit: time limit in second
            bqm: constrained quadratic model
        """
        sampler,qpu_access_time = self.KerberosSampler.sample(self.bqm,max_iter=max_iter,convergence=convergence,qpu_sampler=solver,qpu_params={'label': 'JSSP_bqm_iter'},qpu_reads=num_reads)
        samplerSET=sampler.samples()
        solution=[]
        for key in samplerSET:
            sampler =key
        for key in sampler:
            if sampler[key] !=0 and 'slack' not in key:
                solution.append((key,sampler[key]))
        return solution,qpu_access_time
    def call_bqm_solver_classic(self,convergence,sa_num_reads=1,sa_num_sweeps=10000,subproblem_size=2):#write the related parameters
        """classic solver use the simulated annealing and tabu search parallel and get the best result"""
        sa_num_reads=50
        sa_num_sweeps=10000
        subproblem_size=2
        """iteration = hybrid.RacingBranches(
        hybrid.InterruptableTabuSampler(num_reads=50, tenure=None, timeout=100, initial_states_generator='random'),
        hybrid.EnergyImpactDecomposer(size=subproblem_size)
        | hybrid.SimulatedAnnealingSubproblemSampler(num_reads=sa_num_reads, num_sweeps=sa_num_sweeps,
                 beta_range=None, beta_schedule_type='geometric',
                 initial_states_generator='random')
        | hybrid.SplatComposer()
        ) | hybrid.ArgMin()
        workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=convergence)
        # Solve the problem
        init_state = hybrid.State.from_problem(self.bqm)
        final_state = workflow.run(init_state).result()
        solution=[]
        for key in final_state.samples.first.sample:   #get the solution
            if final_state.samples.first.sample[key] !=0:
                solution.append((key,final_state.samples.first.sample[key]))
        qpu_access_time=0"""
        workflow = hybrid.Parallel(
            hybrid.SimulatedAnnealingProblemSampler(num_reads=50),
            hybrid.TabuProblemSampler(num_reads=50)) | hybrid.MergeSamples()
        init_state = hybrid.State.from_problem(self.bqm)
        final_state = workflow.run(init_state).result()
        solution=[]
        for key in final_state.samples.first.sample:   #get the solution
            if final_state.samples.first.sample[key] !=0:
                solution.append((key,final_state.samples.first.sample[key]))
        qpu_access_time=0
        return solution,qpu_access_time
    def evaluate_solution(self,solution):
        """Find the number of corresponding elements from the solution and calculate its profit"""
        dic={}
        dic["d"]=self.revenue_for_product
        dic["m"]=-self.cost_for_manufacturing_unit
        dic["mu"]=-self.cost_production_of_material
        dic["p"]=-self.cost_for_warehouses
        dic["rm"]=-self.cost_for_warehouses
        dic["s"]=-self.cost_for_supplier
        gain=0

        for element in solution:
            variable=re.search('_(.*)_', element[0])


            if str(variable.group(1))=="d":      #distribution center
                value=element[0].replace("x_d_", "")
                value=int(value)
                amount_d=value

                gain=gain+value*dic[variable.group(1)]-self.cost_for_unmet_customer_demand*(self.initial_customer_demand-value)
            elif str(variable.group(1))=="s" and value < self.initial_suplliers:    #supply, there is canceled supply
                value=element[0].replace("x_s_", "")
                value=int(value)
                amount_s=value

                gain=gain+value*dic[variable.group(1)]-self.cost_for_canceling_supply*(self.initial_suplliers-value)
            elif str(variable.group(1))=="s" and value >= self.initial_suplliers:#supply, there is no canceled supply
                value=element[0].replace("x_s_", "")
                value=int(value)
                amount_s=value

                gain=gain+value*dic[variable.group(1)]
            elif str(variable.group(1))=="mu":#used machines
                value=element[0].replace("x_mu_", "")
                value=int(value)
                gain=gain+value*dic[variable.group(1)]
                amount_mu=value

            elif str(variable.group(1))=="m":#total machines
                value=element[0].replace("x_m_", "")
                value=int(value)
                gain=gain+value*dic[variable.group(1)]
                amount_m=value

            elif str(variable.group(1))=="p":#production
                value=element[0].replace("x_p_", "")
                value=int(value)
                gain=gain+value*dic[variable.group(1)]
                amount_p=value

            elif str(variable.group(1))=="rm":
                value=element[0].replace("x_rm_", "")
                value=int(value)
                gain=gain+value*dic[variable.group(1)]
                amount_rm=value
        """check the constraint for the solution"""
        if amount_p+amount_rm>=self.max_warehouses_stock:
            print('constraint_warehouse fehlt')
        if amount_d!=-amount_p+amount_mu:
            print('constraint_transfer_production fehlt')
        if amount_mu!=amount_s-amount_rm:
            print('constraint_transfer_raw_material fehlt')

        return gain

if __name__== "__main__":
    max_iter,convergence,num_reads=2,2,50
    solver=DWaveSampler()
    a=resilience_computation()
    variables=a.define_variables()
    bqm=a.define_bqm()
    a.constraint_transfer_production(300)#S1:300 S2:300
    a.constraint_transfer_raw_material(300)#S1:300 S2:300
    a.constraint_used_machines(300)#S1:300 S2:300
    a.constraint_variables(1500)#S1:1000 S2:1000 S3:1500
    a.constraint_warehouse(300)#S1:300 S2:300
    profit=a.calculate_initial_profit()
    a.objective_cost(1200/profit)#S1:1000/profit S2:500/profit S3:1200/profit
    #solution,qpu_access_time=a.call_bqm_solver(max_iter,convergence,num_reads,solver)
    solution,qpu_access_time=a.call_bqm_solver_classic(convergence)
    gain=a.evaluate_solution(solution)
    print("New strategy",solution)
    print("Annealing time %s ms"%(qpu_access_time))
    print("Profit with strategy",gain)