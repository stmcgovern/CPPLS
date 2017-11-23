
#ifndef __MATERIAL_DATA_H_INCLUDED__
#define __MATERIAL_DATA_H_INCLUDED__


//keep 0th entry for the basement

#include <vector>

namespace CPPLS
{
using namespace dealii;

class MaterialData
{
public:
    // MaterialData (/*const unsigned int n_rock_types*/);


    const double fluid_density {
        1000
    }; //pore fluid (water) density

    //(TODO:make this weakly dependent on T)
    const double fluid_viscosity {
        0.001
    }; //pore fluid (water) viscosity Pa s

    const double fluid_heat_capacity {
        4180.0
    }; // J/kg K



    double get_heat_capacity (const unsigned int material_id) const
    {
        return heat_capacity[material_id];
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
    const std::vector<double> heat_capacity
    {0.1,1.23,2.01,1.40,2.5, 1.23,2.01,1.40,2.5}; // J kg / K
    const std::vector<double> surface_porosity
    {0,0.61,0.63,0.60,0.61, 0.61,0.61,0.61,0.61,0.61, 0.61,0.61,0.61,0.61,0.65}; // [-]
    const std::vector<double> surface_permeability
    {1e-200,1e-21,3e-21,4e-21,5e-19, 1e-19,3e-18,4e-18,5e-18, 1e-16,3e-18,4e-18,5e-18}; //m^2
    const std::vector<double> solid_density
    {0,2720,2720,2820,3020, 2720,2720,2320,2820,3020, 2720,2720,2320,2820,3020, 2720,2720,2320,2820,3020}; // kg/m^3
    const std::vector<double> compressibility_coefficient
    {1e-200,1e-9,1e-9,1e-9,1e-9, 1e-9,2e-9,1e-8,8e-8, 5e-9,3e-8,1e-8,8e-8}; //Pa^-1

};
}
#endif
