
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
        1024.
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
    double get_thermal_conductivity (const unsigned int material_id) const
    {
        return thermal_conductivity[material_id];
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
    double get_compaction_coefficient (const unsigned int material_id) const
    {
        return compaction_coefficient[material_id];
    }
    double get_depositional_period (const unsigned int material_id) const
    {
        return depositional_periods[material_id];
    }
private:
    const std::vector<double> heat_capacity
    {0,1000,2.01,1.40,2.5, 1.23,2.01,1.40,2.5}; // J/ kg / K
    const std::vector<double> thermal_conductivity
    {0.1,1.23,2.01,1.40,2.5, 1.23,2.01,1.40,2.5}; // W / (m K)
    const std::vector<double> surface_porosity
    {0.61,0.61,0.61,0.60,0.61, 0.61,0.61,0.61,0.61,0.61, 0.61,0.61,0.61,0.61,0.65}; // [-]
    const std::vector<double> surface_permeability
    {1e-300,1e-18,3e-18,4e-21,5e-19, 1e-19,3e-18,4e-18,5e-18, 1e-16,3e-18,4e-18,5e-18}; //m^2
    const std::vector<double> solid_density
    {2720,2720.,2720, 2720,2820,3020, 2720,2720,2320,2820,3020, 2720,2720,2320,2820,3020, 2720,2720,2320,2820,3020}; // kg/m^3
    //This is the compaction coefficient that appears in the compaction law
    const std::vector<double> compaction_coefficient
    {1e-200,8e-8,4e-8,3e-8,5e-8, 1e-9,2e-9,1e-8,8e-8, 5e-9,3e-8,1e-8,8e-8}; //Pa^-1
    const std::vector<double> depositional_periods
    {0,1,0.3,0.125,0.375,9 ,2e-9,1e-8,8e-8, 5e-9,3e-8,1e-8,8e-8}; //Ma [T]

};
}
#endif
