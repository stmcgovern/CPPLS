#ifndef dealii__cppls_utility_functions_h
#define dealii__cppls_utility_functions_h

//#include <deal.II/grid/tria.h>

#include <string>

namespace CPPLS {
using namespace dealii;

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

} // namespace CPPLS

#endif
