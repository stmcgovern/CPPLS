

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/generic_linear_algebra.h>
namespace LA {
using namespace dealii::LinearAlgebraPETSc;
#define USE_PETSC_LA
} // namespace LA

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
//#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>

//#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
//#include <deal.II/numerics/solution_transfer.h>
//#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/filtered_iterator.h>

#include <fstream>
#include <iostream>

#include "level_set_solver.h"
#include "material_data.h"
#include "my_utility_functions.h"
#include "parameters.h"
#include "physical_functions.h"

namespace CPPLS {
using namespace dealii;

constexpr double inflow_rate{-3.15e-11};

template <int dim>
class LayerMovementProblem {
public:
  LayerMovementProblem(const CPPLS::Parameters& parameters, const CPPLS::MaterialData& material_data);
  ~LayerMovementProblem();
  void run();

private:
  // Member Data
  // runtime parameters
  const CPPLS::Parameters parameters;
  const CPPLS::MaterialData material_data;

  // mpi communication
  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  // mesh
  parallel::distributed::Triangulation<dim> triangulation;

  // FE basis space (for P,T, F, and sigma)
  // LS separate

  // pressure
  int degree{2};
  DoFHandler<dim> dof_handler;
  FE_Q<dim> fe;
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  //  // temperature
  //  int degree_T{1};
  //  DoFHandler<dim> dof_handler_T;
  //  FE_Q<dim> fe_T;
  //  IndexSet locally_owned_dofs_T;
  //  IndexSet locally_relevant_dofs_T;

  // level set (can multiple level sets use the same of below? probably not IndexSets)std::vector<IndexSets>
  int degree_LS{2};
  DoFHandler<dim> dof_handler_LS;
  FE_Q<dim> fe_LS;
  IndexSet locally_owned_dofs_LS;
  IndexSet locally_relevant_dofs_LS;

  //  int degree_quad_data{2};
  //  DoFHandler<dim> dof_handler_Q;
  //  FE_DGQ<dim> fe_DGQ_Q;
  //  IndexSet locally_owned_dofs_Q;

  // output stream where only mpi rank 0 output gets to stdout
  ConditionalOStream pcout;

  TimerOutput computing_timer;

  double time_step;
  double current_time;
  double output_number;
  double final_time;
  int timestep_number;

  // set timestepping scheme 1 implicit euler, 1/2 CN, 0 explicit euler
  const double theta;

  ConstraintMatrix constraints_P;
  ConstraintMatrix constraints_T;
  ConstraintMatrix constraints_LS;
  ConstraintMatrix constraints_F;
  ConstraintMatrix constraints_Sigma;

  // FE Field Solution Vectors
  // Ghosted
  // LS
  LA::MPI::Vector locally_relevant_solution_LS_0; // ls
  LA::MPI::Vector old_locally_relevant_solution_LS_0;

  // Pressure
  LA::MPI::Vector locally_relevant_solution_P;
  LA::MPI::Vector old_locally_relevant_solution_P;
  // for use in nonlinear iteration
  LA::MPI::Vector temp_locally_relevant_solution_P;
  LA::MPI::Vector old_temp_locally_relevant_solution_P;

  // Temperature

  LA::MPI::Vector locally_relevant_solution_T;
  LA::MPI::Vector old_locally_relevant_solution_T;

  // Speed function
  LA::MPI::Vector locally_relevant_solution_F;
  LA::MPI::Vector old_locally_relevant_solution_F; // probably not useful
  // this is 0 now
  LA::MPI::Vector locally_relevant_solution_Wxy;

  // Overburden
  LA::MPI::Vector locally_relevant_solution_Sigma;
  LA::MPI::Vector old_locally_relevant_solution_Sigma;
  LA::MPI::Vector temp_locally_relevant_solution_Sigma;

  // Non-ghosted
  LA::MPI::Vector completely_distributed_solution_LS_0;
  LA::MPI::Vector completely_distributed_solution_P;
  LA::MPI::Vector completely_distributed_solution_T;
  LA::MPI::Vector completely_distributed_solution_F;
  // LA::MPI::Vector completely_distributed_solution_Sigma;

  LA::MPI::Vector rhs_P;
  LA::MPI::Vector old_rhs_P;
  LA::MPI::Vector system_rhs_P;

  LA::MPI::Vector rhs_T;
  LA::MPI::Vector old_rhs_T;
  LA::MPI::Vector system_rhs_T;

  LA::MPI::Vector rhs_Sigma;

  LA::MPI::Vector rhs_F;

  // Physical Vectors
  // Compute these algebraic relations on the fly

  //  LA::MPI::Vector bulkdensity;
  //  LA::MPI::Vector porosity;
  //  LA::MPI::Vector old_porosity;
  //  LA::MPI::Vector permeability;

  //  LA::MPI::Vector bulkheat_capacity;
  //  LA::MPI::Vector thermal_conductivity;

  // Sparse Matrices
  LA::MPI::SparseMatrix laplace_matrix_P;
  LA::MPI::SparseMatrix mass_matrix_P;
  LA::MPI::SparseMatrix system_matrix_P;

  LA::MPI::SparseMatrix laplace_matrix_T;
  LA::MPI::SparseMatrix mass_matrix_T;
  LA::MPI::SparseMatrix system_matrix_T;

  LA::MPI::SparseMatrix system_matrix_F;

  LA::MPI::SparseMatrix system_matrix_Sigma;

  // for LS boundary conditions
  std::vector<unsigned int> boundary_values_id_LS;
  std::vector<double> boundary_values_LS;

  // Member Functions

  // create mesh
  void setup_geometry();

  // create fe space
  void setup_dofs();

  // create appropriately sized vectors and matrices
  void setup_system_P();
  void setup_system_T();
  void setup_system_LS();
  void setup_system_F();
  void setup_system_Sigma();

  void initial_conditions();
  void set_boundary_inlet();
  void get_boundary_values_LS(std::vector<unsigned int>& boundary_values_id_LS,
                              std::vector<double>& boundary_values_LS);

  // use level set values to set cell->material_id
  void setup_material_configuration();

  // Pressure
  void assemble_matrices_P();
  void forge_system_P();
  void solve_time_step_P();
  // Temperature
  void assemble_matrices_T();
  void forge_system_T();
  void solve_time_step_T();

  // symbol used for overburden is Sigma
  void assemble_Sigma();
  void solve_Sigma();

  // Speed function (scalar)
  void assemble_F();
  void solve_F();

  double estimate_nl_error();

  void prepare_next_time_step();

  void output_vectors_LS();
  void output_vectors();

  void display_vectors()
  {
    output_vectors_LS();
    output_vectors();
    output_number++;
  }
};

// Constructor

template <int dim>
LayerMovementProblem<dim>::LayerMovementProblem(const CPPLS::Parameters& parameters,
                                                const CPPLS::MaterialData& material_data)
  : parameters(parameters)
  , material_data(material_data)
  , mpi_communicator(MPI_COMM_WORLD)
  , n_mpi_processes{Utilities::MPI::n_mpi_processes(mpi_communicator)}
  , this_mpi_process{Utilities::MPI::this_mpi_process(mpi_communicator)}
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(Triangulation<dim>::smoothing_on_refinement |
                                                             Triangulation<dim>::smoothing_on_coarsening))
  , fe(degree)
  , fe_LS(degree_LS)
  , dof_handler(triangulation)
  , dof_handler_LS(triangulation)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
  , time_step((parameters.stop_time - parameters.start_time) / parameters.n_time_steps)
  , current_time{parameters.start_time}
  , final_time{parameters.stop_time}
  , output_number{0}
  , theta(parameters.theta)

        {};

// Destructor
template <int dim>
LayerMovementProblem<dim>::~LayerMovementProblem()
{
  dof_handler.clear();
  dof_handler_LS.clear();
}

//
template <int dim>
void LayerMovementProblem<dim>::setup_geometry()
{
  TimerOutput::Scope t(computing_timer, "setup_geometry");
  GridGenerator::hyper_cube(triangulation, 0, parameters.box_size, true);
  // GridGenerator::subdivided_hyper_rectangle(triangulation, 0, parameters.box_size);
  triangulation.refine_global(parameters.initial_refinement_level);
  // print_mesh_info(triangulation, "my_grid");
  for (auto cell : filter_iterators(triangulation.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
    cell->set_material_id(0);
  }
}

template <int dim>
void LayerMovementProblem<dim>::setup_dofs()
{
  TimerOutput::Scope t(computing_timer, "setup_dofs");


  dof_handler.distribute_dofs(fe);
  //TODO: put out the sparsity patterns
  Point<dim> direction (0,-1);
  //Not working in parallel now
  //DoFRenumbering::downstream(dof_handler, direction, true);


  pcout << std::endl
        << "============DofHandler===============" << std::endl
        << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl
        << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl
        << std::endl;

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
}

template <int dim>
void LayerMovementProblem<dim>::setup_system_P()
{
  TimerOutput::Scope t(computing_timer, "setup_P");

  pcout << std::endl << "============Pressure===============" << std::endl << std::endl;

  // vector setup
  locally_relevant_solution_P.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  temp_locally_relevant_solution_P.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  old_temp_locally_relevant_solution_P.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  completely_distributed_solution_P.reinit(locally_owned_dofs, mpi_communicator);

  old_locally_relevant_solution_P.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  rhs_P.reinit(locally_owned_dofs, mpi_communicator);
  old_rhs_P.reinit(locally_owned_dofs, mpi_communicator);

  system_rhs_P.reinit(locally_owned_dofs, mpi_communicator);

  // constraints

  constraints_P.clear();

  constraints_P.reinit(locally_relevant_dofs);

  DoFTools::make_hanging_node_constraints(dof_handler, constraints_P);
  // zero dirichlet at top
  VectorTools::interpolate_boundary_values(dof_handler, 3, ZeroFunction<dim>(),
                                           constraints_P); // TODO get rid of raw number
  constraints_P.close();

  // create sparsity pattern

  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_P, false);
  SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.n_locally_owned_dofs_per_processor(), mpi_communicator,
                                             locally_relevant_dofs);
  // setup matrices

  system_matrix_P.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  laplace_matrix_P.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  mass_matrix_P.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
}

template <int dim>
void LayerMovementProblem<dim>::setup_system_T()
{
  TimerOutput::Scope t(computing_timer, "setup_system_T");

  pcout << std::endl << "============Temperature===============" << std::endl << std::endl;

  // vector setup
  locally_relevant_solution_T.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  old_locally_relevant_solution_T.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  completely_distributed_solution_T.reinit(locally_owned_dofs, mpi_communicator);

  rhs_T.reinit(locally_owned_dofs, mpi_communicator);
  old_rhs_T.reinit(locally_owned_dofs, mpi_communicator);
  system_rhs_T.reinit(locally_owned_dofs, mpi_communicator);

  // constraints

  constraints_T.clear();
  constraints_T.reinit(locally_relevant_dofs);

  DoFTools::make_hanging_node_constraints(dof_handler, constraints_T);
  // zero dirichlet at top
  VectorTools::interpolate_boundary_values(dof_handler, 3, ZeroFunction<dim>(),
                                           constraints_T); // TODO again raw number for boundary_id
  // Keep top at fixed temperature, TODO check compatibility condition
  // VectorTools::interpolate_boundary_values(dof_handler_T, 3, ConstantFunction<dim>(20), constraints_T);
  constraints_T.close();

  // create sparsity pattern

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_T, false);
  SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.n_locally_owned_dofs_per_processor(), mpi_communicator,
                                             locally_relevant_dofs);
  // setup matrices

  system_matrix_T.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  laplace_matrix_T.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  mass_matrix_T.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
}

template <int dim>
void LayerMovementProblem<dim>::setup_system_Sigma()
{
  // First of two SUPG problems
  TimerOutput::Scope t(computing_timer, "setup_system_Sigma");

  pcout << std::endl << "============Overburden===============" << std::endl << std::endl;

  // vector setup
  locally_relevant_solution_Sigma.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  old_locally_relevant_solution_Sigma.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  //temp_locally_relevant_solution_Sigma.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  rhs_Sigma.reinit(locally_owned_dofs, mpi_communicator);

  // constraints

  constraints_Sigma.clear();
  constraints_Sigma.reinit(locally_relevant_dofs);

  DoFTools::make_hanging_node_constraints(dof_handler, constraints_Sigma);

  // inflow bc at top
  VectorTools::interpolate_boundary_values(dof_handler, 3, ConstantFunction<dim>(inflow_rate),
                                           constraints_Sigma); // TODO get rid of numbers!
  constraints_Sigma.close();

  // create sparsity pattern

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_Sigma, false);
  SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.n_locally_owned_dofs_per_processor(), mpi_communicator,
                                             locally_relevant_dofs);
  // setup matrix

  system_matrix_Sigma.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
}

template <int dim>
void LayerMovementProblem<dim>::setup_system_F()
{
  // Second of two SUPG problems
  TimerOutput::Scope t(computing_timer, "setup_system_F");

  pcout << std::endl << "============Speed Function===============" << std::endl << std::endl;

  // vector setup
  locally_relevant_solution_F.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  old_locally_relevant_solution_F.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  completely_distributed_solution_F.reinit(locally_owned_dofs, mpi_communicator);

  rhs_F.reinit(locally_owned_dofs, mpi_communicator);

  // constraints

  constraints_F.clear();
  constraints_F.reinit(locally_relevant_dofs);

  DoFTools::make_hanging_node_constraints(dof_handler, constraints_F);

  // inflow bc at top
  VectorTools::interpolate_boundary_values(dof_handler, 3, ConstantFunction<dim>(inflow_rate),
                                           constraints_F); // TODO put in SedimentationRate
  constraints_F.close();

  // create sparsity pattern

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_F, false);
  SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.n_locally_owned_dofs_per_processor(), mpi_communicator,
                                             locally_relevant_dofs);
  // setup matrix

  system_matrix_F.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
}

template <int dim>
void LayerMovementProblem<dim>::setup_system_LS()
{
  // Note: just solution vectors here, no matrices
  TimerOutput::Scope t(computing_timer, "setup_system_LS");

  dof_handler_LS.distribute_dofs(fe_LS);

  pcout << std::endl
        << "============LEVEL SETS===============" << std::endl
        << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl
        << "Number of degrees of freedom: " << dof_handler_LS.n_dofs() << std::endl
        << std::endl;

  locally_owned_dofs_LS = dof_handler_LS.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_LS, locally_relevant_dofs_LS);

  // vector setup
  locally_relevant_solution_LS_0.reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);

  completely_distributed_solution_LS_0.reinit(locally_owned_dofs_LS, mpi_communicator);

  // non-vertical zero vector to feed into LevelSetSolver
  locally_relevant_solution_Wxy.reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);

  locally_relevant_solution_F.reinit(locally_owned_dofs_LS, locally_relevant_dofs_LS, mpi_communicator);
  completely_distributed_solution_F.reinit(locally_owned_dofs_LS, mpi_communicator);

  // constraints

  constraints_LS.clear();

  constraints_LS.reinit(locally_relevant_dofs_LS);

  DoFTools::make_hanging_node_constraints(dof_handler_LS, constraints_LS);

  constraints_LS.close();
}

template <int dim>
void LayerMovementProblem<dim>::initial_conditions()
{
  // Precondition: the non/ghosted vectors have been initialized, and constraints closed (in setup functions)

  // init condition for P (TODO should use call to VectorTools::interpolate)
  completely_distributed_solution_P = 0;
  VectorTools::interpolate_boundary_values(dof_handler, /*top boundary*/ 3, ZeroFunction<dim>(), constraints_P);
  constraints_P.distribute(completely_distributed_solution_P);
  locally_relevant_solution_P = completely_distributed_solution_P;

  // init condition for T   //TODO
  completely_distributed_solution_T = 0;
  VectorTools::interpolate_boundary_values(dof_handler, /*top boundary*/ 3, ZeroFunction<dim>(), constraints_T);
  constraints_T.distribute(completely_distributed_solution_T);
  locally_relevant_solution_T = completely_distributed_solution_T;

  // init condition for LS
  // all the others will share this
  completely_distributed_solution_LS_0 = 0;
  VectorTools::interpolate(dof_handler_LS, Initial_LS<dim>(0.0005, parameters.box_size),
                           completely_distributed_solution_LS_0);
  constraints_LS.distribute(completely_distributed_solution_LS_0);
  locally_relevant_solution_LS_0 = completely_distributed_solution_LS_0;
}

template <int dim>
void LayerMovementProblem<dim>::get_boundary_values_LS(std::vector<unsigned int>& boundary_values_id_LS,
                                                       std::vector<double>& boundary_values_LS)
{
  std::map<unsigned int, double> map_boundary_values_LS;
  unsigned int boundary_id = 0;

  // set_boundary_inlet();
  boundary_id = 10; // inlet
  // we define the inlet to be at the top, i.e. boundary_id=3
  VectorTools::interpolate_boundary_values(dof_handler_LS, 3, BoundaryPhi<dim>(1.0), map_boundary_values_LS);
  boundary_values_id_LS.resize(map_boundary_values_LS.size());
  boundary_values_LS.resize(map_boundary_values_LS.size());
  std::map<unsigned int, double>::const_iterator boundary_value_LS = map_boundary_values_LS.begin();
  for (int i = 0; boundary_value_LS != map_boundary_values_LS.end(); ++boundary_value_LS, ++i) {
    boundary_values_id_LS[i] = boundary_value_LS->first;
    boundary_values_LS[i] = boundary_value_LS->second;
  }
}

template <int dim>
void LayerMovementProblem<dim>::assemble_Sigma()
{
  TimerOutput::Scope t(computing_timer, "assemble_Sigma");
  const AdvectionField<dim> advection_field;
  const QGauss<dim> quadrature_formula(degree + 2);
  const QGauss<dim - 1> face_quadrature_formula(degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_quadrature_points | update_JxW_values | update_gradients);

  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   update_values | update_quadrature_points | update_normal_vectors |
                                       update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> rhs_at_quad(n_q_points);
  std::vector<Tensor<1, dim>> advection_directions(n_q_points);
  std::vector<Tensor<1, dim>> face_advection_directions(n_face_q_points);

  advection_field.value_list(fe_values.get_quadrature_points(), advection_directions);
  std::vector<double> overburden_at_quad(n_q_points);
  std::vector<double> pressure_at_quad(n_q_points);

  Point<dim> point_for_depth;
  SedimentationRate<dim> sedRate;

  std::vector<double> sedimentation_rate(n_q_points);

  Vector<double> cell_rhs(dofs_per_cell);
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  rhs_Sigma = 0;
  system_matrix_Sigma = 0;

  for (auto cell : filter_iterators(dof_handler.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {

    fe_values.reinit(cell);
    fe_values.get_function_values(locally_relevant_solution_P, pressure_at_quad);
    fe_values.get_function_values(locally_relevant_solution_Sigma, overburden_at_quad);

    // TODO consider moving these properties to the quad point level, not just cell level
    const double initial_porosity = material_data.get_surface_porosity(cell->material_id());
    const double compaction_coefficient = material_data.get_compressibility_coefficient(cell->material_id());
    const double rock_density = material_data.get_solid_density(cell->material_id());

    sedRate.value_list(fe_values.get_quadrature_points(), sedimentation_rate, 1);

    cell_rhs = 0;
    cell_matrix = 0;
    const double delta = 0.1 * cell->diameter();

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      point_for_depth = fe_values.quadrature_point(q_point);
      const double hydrostatic = 9.81 * material_data.fluid_density *
                                 (parameters.box_size - point_for_depth[1]); // TODO make this dim independent
      const double phi = porosity(pressure_at_quad[q_point], overburden_at_quad[q_point], initial_porosity,
                                  compaction_coefficient, hydrostatic);
      Assert(0 < hydrostatic, ExcInternalError());
      Assert(0 < phi, ExcInternalError());
      Assert(phi < 1, ExcInternalError());

      const double rho_b = bulkdensity(phi, material_data.fluid_density, rock_density);

      rhs_at_quad[q_point] = 9.81 * rho_b;

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {

          cell_matrix(i, j) += ((advection_directions[q_point] * fe_values.shape_grad(j, q_point) *
                                 (fe_values.shape_value(i, q_point) +
                                  delta * (advection_directions[q_point] * fe_values.shape_grad(i, q_point)))) *
                                fe_values.JxW(q_point));

          cell_rhs(i) += (fe_values.shape_value(i, q_point) +
                          delta * (advection_directions[q_point] * fe_values.shape_grad(i, q_point))) *
                         rhs_at_quad[q_point] * fe_values.JxW(q_point);
        } // end j
      }   // end i
    }     // end q

    // For the inflow boundary term

    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      if (cell->face(face)->at_boundary()) {
        fe_face_values.reinit(cell, face);

        advection_field.value_list(fe_face_values.get_quadrature_points(), face_advection_directions);
        for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
          // the following determines whether inflow or not
          if (fe_face_values.normal_vector(q_point) * face_advection_directions[q_point] < 0)
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) -= (face_advection_directions[q_point] * fe_face_values.normal_vector(q_point) *
                                      fe_face_values.shape_value(i, q_point) * fe_face_values.shape_value(j, q_point) *
                                      fe_face_values.JxW(q_point));
              cell_rhs(i) -=
                  (face_advection_directions[q_point] * fe_face_values.normal_vector(q_point) *
                   sedimentation_rate[q_point] * fe_face_values.shape_value(i, q_point) * fe_face_values.JxW(q_point));
            }
      }

    cell->get_dof_indices(local_dof_indices); // distribute to correct globally numbered vector

    constraints_Sigma.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix_Sigma,
                                                 rhs_Sigma);
  } // end cell loop

  rhs_Sigma.compress(VectorOperation::add);
  system_matrix_Sigma.compress(VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::solve_Sigma()
{
  TimerOutput::Scope t(computing_timer, "solve_Sigma");

  LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(dof_handler.n_dofs(), 1e-12 * rhs_Sigma.l2_norm());
  //  LA::SolverBicgstab solver(solver_control, mpi_communicator);
  LA::SolverGMRES solver(solver_control, mpi_communicator);
  //  LA::MPI::PreconditionAMG preconditioner;
  //  LA::MPI::PreconditionAMG::AdditionalData data;
  //  LA::MPI::PreconditionSSOR preconditioner;
  //  LA::MPI::PreconditionSSOR::AdditionalData data;
  //LA::MPI::PreconditionJacobi preconditioner;
  //LA::MPI::PreconditionJacobi::AdditionalData data;
  //LA::PreconditionSSOR preconditioner;
  //does not compile with this
  //LA::MPI::PreconditionBlockJacobi preconditioner;
  //LA::PreconditionBlockJacobi preconditioner;
  //does with this
  PETScWrappers::PreconditionBlockJacobi preconditioner;
  PETScWrappers::PreconditionBlockJacobi::AdditionalData data;
  //data.symmetric_operator = false;
  preconditioner.initialize(system_matrix_Sigma, data);

  solver.solve(system_matrix_Sigma, completely_distributed_solution, rhs_Sigma, preconditioner);

  pcout << " Overburden supg system solved in " << solver_control.last_step() << " iterations." << std::endl;

  constraints_Sigma.distribute(completely_distributed_solution);

  locally_relevant_solution_Sigma = completely_distributed_solution;
}

template <int dim>
void LayerMovementProblem<dim>::assemble_F()
{

  TimerOutput::Scope t(computing_timer, "assemble_F");
  const AdvectionField<dim> advection_field;
  const QGauss<dim> quadrature_formula(degree + 2);
  const QGauss<dim - 1> face_quadrature_formula(degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_quadrature_points | update_JxW_values | update_gradients);

  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   update_values | update_quadrature_points | update_normal_vectors |
                                       update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> rhs_at_quad(n_q_points);
  std::vector<Tensor<1, dim>> advection_directions(n_q_points);
  std::vector<Tensor<1, dim>> face_advection_directions(n_face_q_points);

  advection_field.value_list(fe_values.get_quadrature_points(), advection_directions);
  std::vector<double> overburden_at_quad(n_q_points);
  std::vector<double> old_overburden_at_quad(n_q_points);
  std::vector<double> pressure_at_quad(n_q_points);
  std::vector<double> old_pressure_at_quad(n_q_points);

  Point<dim> point_for_depth;
  SedimentationRate<dim> sedRate;

  std::vector<double> sedimentation_rate(n_q_points);

  Vector<double> cell_rhs(dofs_per_cell);
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  rhs_F = 0;
  system_matrix_F = 0;

  for (auto cell : filter_iterators(dof_handler.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {

    fe_values.reinit(cell);

    fe_values.get_function_values(locally_relevant_solution_P, pressure_at_quad);
    fe_values.get_function_values(locally_relevant_solution_Sigma, overburden_at_quad);
    fe_values.get_function_values(old_locally_relevant_solution_P, old_pressure_at_quad);
    fe_values.get_function_values(old_locally_relevant_solution_Sigma, old_overburden_at_quad);

    // TODO consider moving these properties to the quad point level, not just cell level
    const double initial_porosity = material_data.get_surface_porosity(cell->material_id());
    const double compaction_coefficient = material_data.get_compressibility_coefficient(cell->material_id());

    sedRate.value_list(fe_values.get_quadrature_points(), sedimentation_rate, 1);

    cell_rhs = 0;
    cell_matrix = 0;
    const double delta = 0.1 * cell->diameter();

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      point_for_depth = fe_values.quadrature_point(q_point);
      const double hydrostatic = 9.81 * material_data.fluid_density *
                                 (parameters.box_size - point_for_depth[1]); // TODO make this dim independent
      Assert(0 < hydrostatic, ExcInternalError());
      const double phi = porosity(pressure_at_quad[q_point], overburden_at_quad[q_point], initial_porosity,
                                  compaction_coefficient, hydrostatic);

      Assert(0 < phi, ExcInternalError());
      Assert(phi < 1, ExcInternalError());

      const double old_phi = porosity(old_pressure_at_quad[q_point], old_overburden_at_quad[q_point], initial_porosity,
                                      compaction_coefficient, hydrostatic);

      Assert(0 < old_phi, ExcInternalError());
      Assert(old_phi < 1, ExcInternalError());

      const double dphidt = (phi - old_phi) / time_step;

      // Assert dphidt <1;
      Assert(dphidt <= 0, ExcInternalError());

      rhs_at_quad[q_point] = dphidt / (1 - phi);

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {

          cell_matrix(i, j) += ((advection_directions[q_point] * fe_values.shape_grad(j, q_point) *
                                 (fe_values.shape_value(i, q_point) +
                                  delta * (advection_directions[q_point] * fe_values.shape_grad(i, q_point)))) *
                                fe_values.JxW(q_point));

          cell_rhs(i) += (fe_values.shape_value(i, q_point) +
                          delta * (advection_directions[q_point] * fe_values.shape_grad(i, q_point))) *
                         rhs_at_quad[q_point] * fe_values.JxW(q_point);
        } // end j
      }   // end i
    }     // end q

    // For the inflow boundary term

    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      if (cell->face(face)->at_boundary()) {
        fe_face_values.reinit(cell, face);

        advection_field.value_list(fe_face_values.get_quadrature_points(), face_advection_directions);
        for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
          // the following determines whether inflow or not
          if (fe_face_values.normal_vector(q_point) * face_advection_directions[q_point] < 0)
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) -= (face_advection_directions[q_point] * fe_face_values.normal_vector(q_point) *
                                      fe_face_values.shape_value(i, q_point) * fe_face_values.shape_value(j, q_point) *
                                      fe_face_values.JxW(q_point));
              cell_rhs(i) -=
                  (face_advection_directions[q_point] * fe_face_values.normal_vector(q_point) *
                   sedimentation_rate[q_point] * fe_face_values.shape_value(i, q_point) * fe_face_values.JxW(q_point));
            }
      }

    cell->get_dof_indices(local_dof_indices); // distribute to correct globally numbered vector

    constraints_F.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix_F, rhs_F);
  } // end cell loop

  rhs_F.compress(VectorOperation::add);
  system_matrix_F.compress(VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::solve_F()
{
  TimerOutput::Scope t(computing_timer, "solve_F");

  LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(dof_handler.n_dofs(), 1e-12 * rhs_F.l2_norm());
  //  LA::SolverBicgstab solver(solver_control, mpi_communicator);
  LA::SolverGMRES solver(solver_control, mpi_communicator);
  //  LA::MPI::PreconditionAMG preconditioner;
  //  LA::MPI::PreconditionAMG::AdditionalData data;
  //  LA::MPI::PreconditionSSOR preconditioner;
  //  LA::MPI::PreconditionSSOR::AdditionalData data;
  LA::MPI::PreconditionJacobi preconditioner;
  LA::MPI::PreconditionJacobi::AdditionalData data;

  // data.symmetric_operator = false;
  preconditioner.initialize(system_matrix_F, data);

  solver.solve(system_matrix_F, completely_distributed_solution, rhs_F, preconditioner);

  pcout << " Speed function system solved in " << solver_control.last_step() << " iterations." << std::endl;

  constraints_F.distribute(completely_distributed_solution);

  locally_relevant_solution_F = completely_distributed_solution;
}


// TODO fold this into P,F,Sigma, T assemblies to assign at quad point level
template <int dim>
void LayerMovementProblem<dim>::setup_material_configuration()
{
  TimerOutput::Scope t(computing_timer, "set_material_configuration");
  // This function sets material ids of cells based on the location of the interface, i.e. loc_rel_solution_LS

  const QGauss<dim> quadrature_formula(degree + 2);

  FEValues<dim> fe_values(fe_LS, quadrature_formula, update_values | update_quadrature_points);

  //  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  //  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> LS0_at_quad(n_q_points);

  // std::vector<double> bulkdensity_at_quad(n_q_points);
  // double eps= GridTools::minimal_cell_diameter(triangulation)/std::sqrt(2);
  //      const double eps=0.001;
  //        double H=0;
  //          // get rho, nu
  //          if (phi>eps)
  //            H=1;
  //          else if (phi<-eps)
  //            H=-1;
  //          else
  //            H=phi/eps;
  //          diff_coeff=1000*(1+H)/2.+10*(1-H)/2.;

  // std::vector<double> id_sum(5);
  double id_sum0{0};

  for (auto cell : filter_iterators(dof_handler_LS.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
    id_sum0 = 0;

    fe_values.reinit(cell);
    fe_values.get_function_values(locally_relevant_solution_LS_0, LS0_at_quad);


    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
//         output_vectors_LS();
//         abort();
        Assert(LS0_at_quad[q_point] < 1.5, ExcInternalError());
        Assert(-1.5 < LS0_at_quad[q_point], ExcInternalError());
        id_sum0 += LS0_at_quad[q_point];
    }
    if (id_sum0 <= 0) {
      cell->set_material_id(0);
    }
    else
      {
        cell->set_material_id(1);
      }
  } // end cell loop
}

template <int dim>
void LayerMovementProblem<dim>::assemble_matrices_P()
{
  TimerOutput::Scope t(computing_timer, "assembly_P");
  const QGauss<dim> quadrature_formula(degree + 2);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_laplace_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  Point<dim> point_for_depth;

  SedimentationRate<dim> sedRate;

  std::vector<double> pressure_at_quad(n_q_points);
  std::vector<double> overburden_at_quad(n_q_points);
  std::vector<double> old_overburden_at_quad(n_q_points);
  std::vector<double> old_pressure_at_quad(n_q_points);

  for (auto cell : filter_iterators(dof_handler.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
    cell_laplace_matrix = 0;
    cell_mass_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit(cell);

    fe_values.get_function_values(locally_relevant_solution_P, pressure_at_quad);
    fe_values.get_function_values(locally_relevant_solution_Sigma, overburden_at_quad);
    fe_values.get_function_values(old_locally_relevant_solution_P, old_pressure_at_quad);
    fe_values.get_function_values(old_locally_relevant_solution_Sigma, old_overburden_at_quad);

    // TODO consider moving these properties to the quad point level, not just cell level
    const double initial_porosity = material_data.get_surface_porosity(cell->material_id());
    const double compaction_coefficient = material_data.get_compressibility_coefficient(cell->material_id());
    const double initial_permeability = material_data.get_surface_permeability(cell->material_id());

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
//        if(0 <= pressure_at_quad[q_point]){
//            output_vectors();
//            abort();
//          }
      Assert( -0.1 <pressure_at_quad[q_point], ExcInternalError());
      Assert(0 < overburden_at_quad[q_point], ExcInternalError());

      point_for_depth = fe_values.quadrature_point(q_point);
      const double hydrostatic = 9.81 * material_data.fluid_density *
                                 (parameters.box_size - point_for_depth[1]); // TODO make this dim independent
      Assert(0 < hydrostatic, ExcInternalError());
      const double phi = porosity(pressure_at_quad[q_point], overburden_at_quad[q_point], initial_porosity,
                                  compaction_coefficient, hydrostatic);
      Assert(0 < phi, ExcInternalError());
      Assert(phi < 1, ExcInternalError());

      const double perm_k = permeability(phi, initial_permeability, initial_porosity);
      // pcout<<"perm_k"<<perm_k<<"init<<"<< initial_permeability<<std::endl;
      Assert(0 < perm_k, ExcInternalError());
      // Assert(perm_k <= initial_permeability, ExcInternalError());

      const double old_phi = porosity(old_pressure_at_quad[q_point], old_overburden_at_quad[q_point], initial_porosity,
                                      compaction_coefficient, hydrostatic);
      Assert(0 < old_phi, ExcInternalError());
      Assert(old_phi < 1, ExcInternalError());

      const double dphidt = (phi - old_phi) / time_step;
      Assert(dphidt <= 0, ExcInternalError());

      const double diff_coeff_at_quad = (perm_k / material_data.fluid_viscosity);
      const double rhs_coeff = material_data.get_compressibility_coefficient(cell->material_id()) * phi / (1 - phi);
      const double rhs_at_quad = rhs_coeff * (overburden_at_quad[q_point] - old_overburden_at_quad[q_point]) / time_step -
         9.8 * material_data.fluid_density * (inflow_rate); //TODO replace with sedRate

      Assert (0 <= rhs_at_quad, ExcInternalError());

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {

          cell_laplace_matrix(i, j) += diff_coeff_at_quad * (fe_values.shape_grad(i, q_point) *
                                                             fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));

          cell_mass_matrix(i, j) +=
              (fe_values.shape_value(i, q_point) * fe_values.shape_value(j, q_point) * fe_values.JxW(q_point));

          cell_rhs(i) += (rhs_at_quad * fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));
        }
      }
    } // end q

    cell->get_dof_indices(local_dof_indices);
    constraints_P.distribute_local_to_global(cell_laplace_matrix, cell_rhs, local_dof_indices, laplace_matrix_P, rhs_P);
    constraints_P.distribute_local_to_global(cell_mass_matrix, local_dof_indices, mass_matrix_P);
  } // end cell

  laplace_matrix_P.compress(VectorOperation::add);
  mass_matrix_P.compress(VectorOperation::add);
  rhs_P.compress(VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::forge_system_P()
{
  // in this function we manipulate A, M, F, resulting from assemble_matrices_P
  TimerOutput::Scope t(computing_timer, "forge_P");
  LA::MPI::Vector tmp;
  LA::MPI::Vector forcing_terms;

  tmp.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  forcing_terms.reinit(locally_owned_dofs, mpi_communicator);

  old_locally_relevant_solution_P = locally_relevant_solution_P;
  mass_matrix_P.vmult(system_rhs_P, old_locally_relevant_solution_P);

  laplace_matrix_P.vmult(tmp, old_locally_relevant_solution_P);
  //  pcout << "laplace symmetric: " << laplace_matrix.is_symmetric()<<std::endl;
  system_rhs_P.add(-(1 - theta) * time_step, tmp);

  forcing_terms.add(time_step * theta, rhs_P);

  forcing_terms.add(time_step * (1 - theta), old_rhs_P);

  system_rhs_P += forcing_terms;
  // system_matrix.compress (VectorOperation::add);

  system_matrix_P.copy_from(mass_matrix_P);
  // system_matrix.compress (VectorOperation::add);

  system_matrix_P.add(laplace_matrix_P, time_step * (1 - theta));
}

template <int dim>
void LayerMovementProblem<dim>::solve_time_step_P()
{
  TimerOutput::Scope t(computing_timer, "solve_time_step_P");

  LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(dof_handler.n_dofs(), 1e-12 * system_rhs_P.l2_norm());
  LA::SolverCG solver(solver_control, mpi_communicator);

  LA::MPI::PreconditionAMG preconditioner;

  LA::MPI::PreconditionAMG::AdditionalData data;

  data.symmetric_operator = true;
  preconditioner.initialize(system_matrix_P, data);

  solver.solve(system_matrix_P, completely_distributed_solution, system_rhs_P, preconditioner);

  pcout << " Pressure system solved in " << solver_control.last_step() << " iterations." << std::endl;

  constraints_P.distribute(completely_distributed_solution);

  locally_relevant_solution_P = completely_distributed_solution;
}

template <int dim>
void LayerMovementProblem<dim>::assemble_matrices_T()
{
  TimerOutput::Scope t(computing_timer, "assembly_T");
  const QGauss<dim> quadrature_formula(degree + 2);
  const QGauss<dim - 1> face_quadrature_formula(degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   update_values | update_quadrature_points | update_normal_vectors |
                                       update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  FullMatrix<double> cell_laplace_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  Point<dim> point_for_depth;

  std::vector<double> pressure_at_quad(n_q_points);
  std::vector<double> overburden_at_quad(n_q_points);
  std::vector<double> bulkheat_capacity_at_quad(n_q_points);
  std::vector<double> thermal_conductivity_at_quad(n_q_points);

  for (auto cell : filter_iterators(dof_handler.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
    cell_laplace_matrix = 0;
    cell_mass_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit(cell);
    fe_values.get_function_values(locally_relevant_solution_P, pressure_at_quad);
    fe_values.get_function_values(locally_relevant_solution_Sigma, overburden_at_quad);

    // TODO consider moving these properties to the quad point level, not just cell level
    const double initial_porosity = material_data.get_surface_porosity(cell->material_id());
    const double compaction_coefficient = material_data.get_compressibility_coefficient(cell->material_id());
    const double rock_density = material_data.get_solid_density(cell->material_id());
    const double heat_capacity = material_data.get_heat_capacity(cell->material_id());

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      point_for_depth = fe_values.quadrature_point(q_point);
      const double hydrostatic = 9.81 * material_data.fluid_density *
                                 (parameters.box_size - point_for_depth[1]); // TODO make this dim independent
      const double phi = porosity(pressure_at_quad[q_point], overburden_at_quad[q_point], initial_porosity,
                                  compaction_coefficient, hydrostatic);
      Assert(0 < phi < 1, ExcInternalError());

      const double rho_b = bulkdensity(phi, material_data.fluid_density, rock_density);
      const double bulk_hc = bulkheatcapacity(phi, material_data.fluid_heat_capacity, heat_capacity);

      //      fe_values.get_function_values(thermal_conductivity, thermal_conductivity_at_quad);
      // TODO
      const double diff_coeff_at_quad = 10;
      //  thermal_conductivity_at_quad[q_point] /(bulkheat_capacity_at_quad[q_point]*bulkdensity_at_quad[q_point] );
      const double rhs_at_quad = 0; // TODO bottom boundary flux from parameter file

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {

          cell_laplace_matrix(i, j) += diff_coeff_at_quad * fe_values.shape_grad(i, q_point) *
                                       fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);

          cell_mass_matrix(i, j) += (rho_b * bulk_hc * fe_values.shape_value(i, q_point) *
                                     fe_values.shape_value(j, q_point) * fe_values.JxW(q_point));

          //          cell_rhs(i) += (right_hand_side.value(fe_values.quadrature_point(q_point)) *
          //                          fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));
          cell_rhs(i) += (rhs_at_quad * fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));
        } // end j
      }   // end i
    }     // end q

    for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number) {
      if (cell->face(face_number)->at_boundary() &&
          (cell->face(face_number)->boundary_id() == 2)) // bottom of domain TODO remove raw number
      {
        fe_face_values.reinit(cell, face_number);
        for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
          //                const double neumann_value
          //                  = (exact_solution.gradient (fe_face_values.quadrature_point(q_point)) *
          //                     fe_face_values.normal_vector(q_point));
          // TODO pick the right flux value
          const double neumann_value = 100; // represents the bottom flux boundary condition
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            cell_rhs(i) += (neumann_value * fe_face_values.shape_value(i, q_point) * fe_face_values.JxW(q_point));
          }
        }
      }
    } // end face loop

    cell->get_dof_indices(local_dof_indices);
    constraints_T.distribute_local_to_global(cell_laplace_matrix, cell_rhs, local_dof_indices, laplace_matrix_T, rhs_T);
    constraints_T.distribute_local_to_global(cell_mass_matrix, local_dof_indices, mass_matrix_T);
  } // end cell

  laplace_matrix_T.compress(VectorOperation::add);
  mass_matrix_T.compress(VectorOperation::add);
  rhs_T.compress(VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::forge_system_T()
{
  // in this function we manipulate A, M, F, resulting from assemble_matrices_T
  TimerOutput::Scope t(computing_timer, "forge_T");
  LA::MPI::Vector tmp;
  LA::MPI::Vector forcing_terms;

  tmp.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  forcing_terms.reinit(locally_owned_dofs, mpi_communicator);

  old_locally_relevant_solution_T = locally_relevant_solution_T;
  mass_matrix_T.vmult(system_rhs_T, old_locally_relevant_solution_T);

  laplace_matrix_T.vmult(tmp, old_locally_relevant_solution_T);
  //  pcout << "laplace symmetric: " << laplace_matrix.is_symmetric()<<std::endl;
  system_rhs_T.add(-(1 - theta) * time_step, tmp);

  forcing_terms.add(time_step * theta, rhs_T);

  forcing_terms.add(time_step * (1 - theta), old_rhs_T);

  system_rhs_T += forcing_terms;
  // system_matrix.compress (VectorOperation::add);

  system_matrix_T.copy_from(mass_matrix_T);
  // system_matrix.compress (VectorOperation::add);

  system_matrix_T.add(laplace_matrix_T, time_step * (1 - theta));
}

template <int dim>
void LayerMovementProblem<dim>::solve_time_step_T()
{
  TimerOutput::Scope t(computing_timer, "solve_time_step_T");

  LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(dof_handler.n_dofs(), 1e-12 * system_rhs_T.l2_norm());
  LA::SolverCG solver(solver_control, mpi_communicator);

  LA::MPI::PreconditionAMG preconditioner;

  LA::MPI::PreconditionAMG::AdditionalData data;

  data.symmetric_operator = true;
  preconditioner.initialize(system_matrix_T, data);

  solver.solve(system_matrix_T, completely_distributed_solution, system_rhs_T, preconditioner);

  pcout << " Temperature system solved in " << solver_control.last_step() << " iterations." << std::endl;

  constraints_T.distribute(completely_distributed_solution);

  locally_relevant_solution_T = completely_distributed_solution;
}

template <int dim>
void LayerMovementProblem<dim>::prepare_next_time_step()
{
  //  old_porosity = porosity;
  //  old_overburden = overburden;
}

template <int dim>
void LayerMovementProblem<dim>::output_vectors_LS()
{
  TimerOutput::Scope t(computing_timer, "output_LS");
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_LS);
  data_out.add_data_vector(locally_relevant_solution_LS_0, "LS");
  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  const std::string filename = ("sol_LS_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                                Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
      filenames.push_back("sol_LS_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                          Utilities::int_to_string(i, 4) + ".vtu");

    std::ofstream master_output(("sol_LS_vectors-" + Utilities::int_to_string(output_number, 3) + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}

// TODO output material id

template <int dim>
void LayerMovementProblem<dim>::output_vectors()
{
  TimerOutput::Scope t(computing_timer, "output");
  // output_number++;
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(locally_relevant_solution_P, "P");
  data_out.add_data_vector(old_locally_relevant_solution_P, "old_P");
  data_out.add_data_vector(locally_relevant_solution_T, "T");
  data_out.add_data_vector(old_locally_relevant_solution_T, "old_T");
  data_out.add_data_vector(locally_relevant_solution_Sigma, "Sigma");
  data_out.add_data_vector(old_locally_relevant_solution_Sigma, "old_Sigma");
  data_out.add_data_vector(locally_relevant_solution_F, "F");
//  data_out.add_data_vector(system_rhs_P, "s_rhs_P");
//  data_out.add_data_vector(rhs_F, "rhs_F" );
//  data_out.add_data_vector(rhs_Sigma, "rhs_s");

  //  LA::MPI::Vector material_kind;
  //  material_kind.reinit(locally_owned_dofs, mpi_communicator);
  //  material_kind = 0;

  //  int i = 0;
  //  for (auto cell : filter_iterators(triangulation.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
  //    material_kind(i) = static_cast<float>(cell->material_id());
  //    ++i;
  //  }
  //  data_out.add_data_vector(material_kind, "material_kind");

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  const std::string filename = ("sol_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                                Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
      filenames.push_back("sol_vectors-" + Utilities::int_to_string(output_number, 3) + "." +
                          Utilities::int_to_string(i, 4) + ".vtu");

    std::ofstream master_output(("sol_vectors-" + Utilities::int_to_string(output_number, 3) + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}



template <int dim>
void LayerMovementProblem<dim>::run()
{
  // common mesh
  setup_geometry();
  // common dofhandler, except for LS
  setup_dofs();
  // these are the 4 systems treated in this code
  setup_system_P();
  setup_system_T();
  setup_system_Sigma();
  setup_system_F();
  // the solution of this system is done in the LevelSetSolver class
  setup_system_LS();

  initial_conditions();

  //const SedimentationRate SedRate(parameters);
  const double SedRate = 3.15e-11;
  // initialize level set solver
  // we use some hardcode defaults for now

  const double min_h = GridTools::minimal_cell_diameter(triangulation) / std::sqrt(2);
  const double cfl = 0.5;
  const double umax = SedRate;  //max_sedRate
  time_step = cfl * min_h / umax;
  // pcout<<"min_h"<<min_h;

  const double cK = 1.0;
  const double cE = 1.0;
  const bool verbose = true;
  std::string ALGORITHM = "MPP_uH";
  const unsigned int TIME_INTEGRATION = 1; // corresponds to SSP33
  LevelSetSolver<dim> level_set_solver0(degree_LS, degree, time_step, cK, cE, verbose, ALGORITHM, TIME_INTEGRATION,
                                        triangulation, mpi_communicator);

  // BOUNDARY CONDITIONS FOR PHI
  get_boundary_values_LS(boundary_values_id_LS, boundary_values_LS);
  level_set_solver0.set_boundary_conditions(boundary_values_id_LS, boundary_values_LS);

  // set INITIAL CONDITION within TRANSPORT PROBLEM
  level_set_solver0.initial_condition(locally_relevant_solution_LS_0, locally_relevant_solution_Wxy,
                                      locally_relevant_solution_F);

  // For the first step, F is just the sedimentation rate

  locally_relevant_solution_F = -1*SedRate;

  //  // TIME STEPPING
  timestep_number = 1;
  for (double time = time_step; time <= final_time; time += time_step, ++timestep_number) {
    pcout << "Time step " << timestep_number << " at t=" << time << std::endl;

    // Solve for F the scalar speed function which is passed to the LevelSetSolver
    // which expects the vector (wx,wy) or (wx,wy,wz)
    // dim=2 we pass F as wy and dim=3 we pass F as wz with 0 otherwise

    // Level set computation
    // original level_set_solver.set_velocity(locally_relevant_solution_u, locally_relevant_solution_v);

    {
      TimerOutput::Scope t(computing_timer, "LS");

      level_set_solver0.set_velocity(locally_relevant_solution_Wxy, locally_relevant_solution_F);
      level_set_solver0.nth_time_step();
      level_set_solver0.get_unp1(locally_relevant_solution_LS_0); // exposes interface vector
    }
    output_vectors();
    // set material ids based on locally_relevant_solution_LS
    setup_material_configuration(); // TODO: move away from cell id to values at quad points

      // First get an overburden solution with the current porosity
      assemble_Sigma();
      solve_Sigma();    // generates l_r_s_Sigma

      // pressure solution
      assemble_matrices_P();
      forge_system_P();
      solve_time_step_P(); // temp_loc_r_s_P

    // Solve temperature (coefficients depend on porosity, and TODO: should influence viscosity)

    //    assemble_matrices_T();
    //    forge_system_T();
    //    solve_time_step_T();

    assemble_F();
    solve_F();

    //    if (get_output && time - (output_number)*output_time > 0)
    //      output_results();
    if (timestep_number % 1 == 0) {
      display_vectors();
    }
    // output_vectors_Q();
    prepare_next_time_step();
  } // end of time loop
}

} // end namespace CPPLS

constexpr int dim{2};
//constexpr double inflow_rate{3.15e-11};

int main(int argc, char* argv[])
{
  // One of the new features in C++11 is the <code>chrono</code> component of
  // the standard library. This gives us an easy way to time the output.
  try {
    using namespace dealii;
    using namespace CPPLS;

    auto t0 = std::chrono::high_resolution_clock::now();

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    CPPLS::Parameters parameters;
    parameters.read_parameter_file("parameters.prm");
    CPPLS::MaterialData material_data;
    {
      LayerMovementProblem<dim> run_layers(parameters, material_data);
      run_layers.run();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      std::cout << "time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                << " milliseconds." << std::endl;
    }
  }
  catch (std::exception& exc) {
    std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  }
  catch (...) {
    std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
