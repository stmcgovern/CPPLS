

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
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

//#include "my_utility_functions.h"
#include "material_data.h"
#include "parameters.h"

namespace CPPLS {
using namespace dealii;

// some free functions

template <int dim>
void print_mesh_info(const Triangulation<dim>& triangulation, const std::string& filename) {
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << triangulation.n_active_cells() << std::endl;

  std::map<unsigned int, unsigned int> boundary_count;
  typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(), endc = triangulation.end();
  for (; cell != endc; ++cell) {
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
      if (cell->face(face)->at_boundary())
        boundary_count[cell->face(face)->boundary_id()]++;
    }
  }
  std::cout << " boundary indicators: ";
  for (std::map<unsigned int, unsigned int>::iterator it = boundary_count.begin(); it != boundary_count.end(); ++it) {
    std::cout << it->first << "(" << it->second << " times) ";
  }
  std::cout << std::endl;

  std::ofstream out(filename.c_str());
  GridOut grid_out;
  grid_out.write_eps(triangulation, out);
  std::cout << " written to " << filename << std::endl << std::endl;
}

double porosity(const double pressure, const double overburden, const double initial_porosity,
                const double compaction_coefficient) {
  return (initial_porosity * std::exp(-1 * compaction_coefficient * (overburden - pressure)));
}
double permeability(const double porosity, const double initial_permeability, const double initial_porosity) {
  return (initial_permeability * (1 - porosity) / (1 - initial_porosity));
}

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
  // FE basis spaces
  // pressure
  int degree_P{1};
  DoFHandler<dim> dof_handler_P;
  FE_Q<dim> fe_P;
  IndexSet locally_owned_dofs_P;
  IndexSet locally_relevant_dofs_P;
  // temperature
  int degree_T{1};
  DoFHandler<dim> dof_handler_T;
  FE_Q<dim> fe_T;
  IndexSet locally_owned_dofs_T;
  IndexSet locally_relevant_dofs_T;
  // level set (can multiple level sets use the same of below? probably not IndexSets)std::vector<IndexSets>
  int degree_LS{1};
  DoFHandler<dim> dof_handler_LS;
  FE_Q<dim> fe_LS;
  IndexSet locally_owned_dofs_LS;
  IndexSet locally_relevant_dofs_LS;

  // output stream where only mpi rank 0 output gets to stdout
  ConditionalOStream pcout;

  TimerOutput computing_timer;

  const double time_step;
  double current_time;
  double output_number;
  double final_time;

  ConstraintMatrix constraints_P;
  ConstraintMatrix constraints_T;
  ConstraintMatrix constraints_V;


  // FE Field Solution Vectors

  LA::MPI::Vector locally_relevant_solution_u; // ls
  LA::MPI::Vector locally_relevant_solution_p;
  LA::MPI::Vector locally_relevant_solution_t;
  LA::MPI::Vector locally_relevant_solution_f; // speed function
  LA::MPI::Vector completely_distributed_solution_u;
  LA::MPI::Vector completely_distributed_solution_p;
  LA::MPI::Vector completely_distributed_solution_t;
  LA::MPI::Vector completely_distributed_solution_f;

  LA::MPI::Vector rhs_p;
  LA::MPI::Vector rhs_t;


  // Physical Vectors

  LA::MPI::Vector overburden;
  LA::MPI::Vector old_overburden;
  LA::MPI::Vector bulkdensity;
  LA::MPI::Vector porosity;
  LA::MPI::Vector permeability;

  //Sparse Matrices
  LA::MPI::SparseMatrix laplace_matrix;
  LA::MPI::SparseMatrix mass_matrix;



  // for boundary conditions
  std::vector<unsigned int> boundary_values_id_u;
  std::vector<unsigned int> boundary_values_id_p;
  std::vector<unsigned int> boundary_values_id_t;
  std::vector<double> boundary_values_u;
  std::vector<double> boundary_values_p;
  std::vector<double> boundary_values_t;


  // Member Functions

  // create mesh
  void setup_geometry();
  void set_boundary_initial_conditions_P();
  void set_boundary_initiald_conditions_T();

  void setup_material_configuration(LA::MPI::Vector /*std::vector<LA::MPI::Vector>*/);
  // initialize vectors
  void setup_dofs_P();
  void setup_dofs_T();

  void assemble_system_P();
  void assemble_system_T();

  void compute_bulkdensity();
  void compute_overburden();
  void compute_porosity();
  void compute_speed_function();

  void output_vectors();
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
  , triangulation(triangulation)
  , fe_P(degree_P)
  , fe_T(degree_T)
  , fe_LS(degree_LS)
  , time_step(0.1)
  , dof_handler_LS(triangulation)
  , dof_handler_P(triangulation)
  , dof_handler_T(triangulation)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)

        {};

//
template <int dim>
void LayerMovementProblem<dim>::setup_geometry() {
  GridGenerator::hyper_cube(triangulation, 0, parameters.box_size, true);
  // GridGenerator::subdivided_hyper_rectangle(triangulation, 0, parameters.box_size);
  triangulation.refine_global(parameters.initial_refinement_level);
  print_mesh_info(triangulation, "my_grid");
}

//template<int dim>
//  void LayerMovementProblem<dim>::setup_system()
//  {
//    TimerOutput::Scope t(computing_timer, "setup");
//    dof_handler.distribute_dofs(fe);

//    pcout << std::endl
//              << "==========================================="
//              << std::endl
//              //<< "Number of active cells: " << triangulation.n_global_active_cells()
//              << std::endl
//              << "Number of degrees of freedom: " << dof_handler.n_dofs()
//              << std::endl
//              << std::endl;

//    locally_owned_dofs = dof_handler.locally_owned_dofs ();
//    DoFTools::extract_locally_relevant_dofs (dof_handler,
//                                             locally_relevant_dofs);

//    //vector setup
//    locally_relevant_solution.reinit (locally_owned_dofs,
//                                      locally_relevant_dofs, mpi_communicator);
//    old_locally_relevant_solution.reinit (locally_owned_dofs,
//                                      locally_relevant_dofs, mpi_communicator);
//    system_rhs.reinit (locally_owned_dofs, mpi_communicator);
//    forcing.reinit(locally_owned_dofs, mpi_communicator);
//    old_forcing.reinit (locally_owned_dofs, mpi_communicator);





//    //constrainst

//    //constraints.clear ();

//    constraints.reinit (locally_relevant_dofs);

//    DoFTools::make_hanging_node_constraints (dof_handler,
//                                             constraints);
//    constraints.close();

//    //create sparsity pattern

//    DynamicSparsityPattern dsp (locally_relevant_dofs);

//    DoFTools::make_sparsity_pattern (dof_handler, dsp,
//                                     constraints, false);
//    SparsityTools::distribute_sparsity_pattern (dsp,
//                                                dof_handler.n_locally_owned_dofs_per_processor(),
//                                                mpi_communicator,
//                                                locally_relevant_dofs);
//    //setup matrices

//    system_matrix.reinit (locally_owned_dofs,
//                          locally_owned_dofs,
//                          dsp,
//                          mpi_communicator);
//    laplace_matrix.reinit (locally_owned_dofs,
//                          locally_owned_dofs,
//                          dsp,
//                          mpi_communicator);
//    mass_matrix.reinit (locally_owned_dofs,
//                          locally_owned_dofs,
//                          dsp,
//                          mpi_communicator);

//  }






template <int dim>
void LayerMovementProblem<dim>::setup_dofs_P() {

  // init condition for p
  completely_distributed_solution_p = 0;
  VectorTools::interpolate(dof_handler_P, /*top boundary*/ 1, ZeroFunction<dim>(), completely_distributed_solution_p);
  constraints_P.distribute(completely_distributed_solution_p);
  locally_relevant_solution_p = completely_distributed_solution_p;
}

template <int dim>
void LayerMovementProblem<dim>::compute_bulkdensity() {
  // temporary
  // const double porosity_simple{0.5};

  FE_Q<dim> fe(1);
  const QGauss<dim> quadrature_formula(3);

  FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  // FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  // Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<double> bulkdensity_at_quad(n_q_points);
  std::vector<double> porosity_at_quad(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  constraints_V.close();

  for (auto cell : filter_iterators(triangulation.active_cell_iterators(), IteratorFilters::LocallyOwnedCell())) {
    fe_values.reinit(cell);
    // query the local porosity - use constant for now
    const double rock_density = material_data.get_solid_density(cell->material_id());

     fe_values.get_function_values(porosity,porosity_at_quad);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      bulkdensity_at_quad[q_point] =
          porosity_at_quad[q_point] * material_data.fluid_density + (1 - porosity_at_quad[q_point]) * rock_density;
    }
    cell->get_dof_indices(local_dof_indices);
    constraints_V.distribute_local_to_global(bulkdensity_at_quad, local_dof_indices, bulkdensity);
  }
  bulkdensity.compress(VectorOperation::add);

}

template <int dim>
void LayerMovementProblem<dim>::compute_overburden() {

  FE_Q<dim> fe(1);
  const QGauss<dim> quadrature_formula(3);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_quadrature_points );

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();



  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> overburden_at_quad(n_q_points);
  std::vector<double> bulkdensity_at_quad(n_q_points);

  Point<dim> point_for_depth;

  for (auto cell : filter_iterators(triangulation.active_cell_iterators(), IteratorFilters::LocallyOwnedCell()))
    {

      fe_values.reinit(cell);

    //fe_values.get_function_values(interface_phi, phi_at_quad);
       fe_values.get_function_values(bulkdensity,bulkdensity_at_quad);
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        point_for_depth=fe_values.quadrature_point(q_point);
        overburden_at_quad[q_point]= /* material_data.g */ 9.81* point_for_depth[2] *bulkdensity_at_quad[q_point];
      }
      cell->get_dof_indices(local_dof_indices);
      constraints_V.distribute_local_to_global(overburden_at_quad, local_dof_indices, overburden);
    }
    overburden.compress(VectorOperation::add);

}
template <int dim>
void LayerMovementProblem<dim>::setup_material_configuration(LinearAlgebraPETSc::MPI::Vector locally_relevant_u)
{

}

template <int dim>
void LayerMovementProblem<dim>::assemble_system_P() {
  TimerOutput::Scope t(computing_timer, "assembly_P");
  const QGauss<dim> quadrature_formula(3);

  // RightHandSide<dim> right_hand_side;
  // //set time too
  // right_hand_side.set_time(time);
  // DiffusionCoefficient<dim> diffusion_coeff;

  FEValues<dim> fe_values(fe_P, quadrature_formula,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe_P.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_laplace_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  //std::vector<double> u_at_quad(n_q_points);
  std::vector<double> porosity_at_quad(n_q_points);
  std::vector<double> overburden_at_quad(n_q_points);
  std::vector<double> old_overburden_at_quad(n_q_points);
  std::vector<double> permeability_at_quad(n_q_points);


  for (auto cell : filter_iterators(dof_handler_P.active_cell_iterators(),
                                      IteratorFilters::LocallyOwnedCell()))
  {
    cell_laplace_matrix = 0;
    cell_mass_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit(cell);
    fe_values.get_function_values(porosity, porosity_at_quad);
    fe_values.get_function_values(permeability, permeability_at_quad);
    fe_values.get_function_values(overburden, overburden_at_quad);
    fe_values.get_function_values(old_overburden, old_overburden_at_quad);


    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
     const double diff_coeff_at_quad = (permeability_at_quad[q_point]/ (porosity_at_quad[q_point]*material_data.fluid_viscosity));
     const double rhs_at_quad = (overburden[q_point]-old_overburden_at_quad[q_point]) / time_step;

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {

          cell_laplace_matrix(i, j) += diff_coeff_at_quad * (fe_values.shape_grad(i, q_point) *
                                                     fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));

          //                    cell_laplace_matrix(i,j) += diffusion_coeff.value(fe_values.quadrature_point
          //                    (q_point)) *
          //                                        (fe_values.shape_grad(i,q_point) *
          //                                         fe_values.shape_grad(j,q_point) *
          //                                         fe_values.JxW(q_point));

          cell_mass_matrix(i, j) +=
              (fe_values.shape_value(i, q_point) * fe_values.shape_value(j, q_point) * fe_values.JxW(q_point));

//          cell_rhs(i) += (right_hand_side.value(fe_values.quadrature_point(q_point)) *
//                          fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));
          cell_rhs(i) += (rhs_at_quad *
                          fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));
        }
      }
    }

    cell->get_dof_indices(local_dof_indices);
    constraints_P.distribute_local_to_global(cell_laplace_matrix, cell_rhs, local_dof_indices, laplace_matrix, rhs_p);
    constraints_P.distribute_local_to_global(cell_mass_matrix, local_dof_indices, mass_matrix);

        }

    // Notice that the assembling above is just a local operation. So, to
    // form the "global" linear system, a synchronization between all
    // processors is needed. This could be done by invoking the function
    // compress(). See @ref GlossCompress  "Compressing distributed objects"
    // for more information on what is compress() designed to do.
    laplace_matrix.compress (VectorOperation::add);
    mass_matrix.compress (VectorOperation::add);
    rhs_p.compress (VectorOperation::add);
}

template <int dim>
void LayerMovementProblem<dim>::output_vectors() {
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_LS);
  data_out.add_data_vector(locally_relevant_solution_u, "u");
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

    std::ofstream master_output((filename + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}

template <int dim>
void LayerMovementProblem<dim>::run() {
  // common mesh
  setup_geometry();
  //
  // initialize level set solver
  // LevelSetSolver<dim> level_set_solver;

  // initialize pressure solver
  // PressureEquation<dim> pressure_solver;
  // initialize temperature solver
  // TemperatureEquation<dim> temperature_solver;

//  // TIME STEPPING
//  for (double timestep_number = 1, time = time_step; time <= final_time; time += time_step, ++timestep_number) {
//    pcout << "Time step " << timestep_number << " at t=" << time << std::endl;

//    state.set_phi(locally_relevant_solution_phi);
//    state.nth_time_step();
//    state.get_velocity(locally_relevant_solution_u, locally_relevant_solution_v);
//    transport_solver.set_velocity(locally_relevant_solution_u, locally_relevant_solution_v);
//    // GET LEVEL SET SOLUTION
//    transport_solver.nth_time_step();
//    transport_solver.get_unp1(locally_relevant_solution_phi); // exposes interface vector
//    if (get_output && time - (output_number)*output_time > 0)
//      output_results();
//  }
}

} // end namespace CPPLS

constexpr int dim{3};

int main(int argc, char* argv[]) {
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

    LayerMovementProblem<dim> run_layers(parameters, material_data);
    run_layers.run();

    auto t1 = std::chrono::high_resolution_clock::now();
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      std::cout << "time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                << " milliseconds." << std::endl;
    }

  } catch (std::exception& exc) {
    std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  } catch (...) {
    std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
