
#ifndef __MATERIAL_DATA_H_INCLUDED__
#define __MATERIAL_DATA_H_INCLUDED__

//For now define only 4 rock types
//keep 0th entry for the basement

#include <vector>

namespace CPPLS
{
  using namespace dealii;

    class MaterialData
      {
      public:
       // MaterialData (/*const unsigned int n_rock_types*/);


        const double fluid_density{1000}; //pore fluid (water) density

        //(TODO:make this weakly dependent on T)
        const double fluid_viscosity{1000}; //pore fluid (water) viscosity Pa s





        double get_heat_conductivity (const unsigned int material_id) const
        {
          return heat_conductivity[material_id];
        }
        double get_surface_porosity (const unsigned int material_id) const
        {
          return surface_porosity[material_id];
        }
        double get_surface_permeability (const unsigned int material_id) const
        {
          return surface_permeability[material_id];
        }
        double get_solid_density (const unsigned int material_id) const
        {
          return solid_density[material_id];
        }
        double get_compressibility_coefficient (const unsigned int material_id) const
        {
          return compressibility_coefficient[material_id];
        }
      private:
        const std::vector<double> heat_conductivity{0,1.23,2.01,1.40,2.5}; // SI is J m-1 s-1 K-1 or W m-1 K-1.
        const std::vector<double> surface_porosity{0,0.61,0.53,0.4,0.65}; // [-]
        const std::vector<double> surface_permeability{0,2e-18,3e-18,4e-18,5e-18}; //m^2
        const std::vector<double> solid_density{2720,2320,2620,2820,3020};// kg/m^3
        const std::vector<double> compressibility_coefficient{5e-8,2e-8,3e-8,1e-8,8e-8}; //Pa^-1

      };
}

#endif 
