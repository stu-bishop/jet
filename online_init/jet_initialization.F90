!> A template of a user to code up customized initial conditions.
module jet_initialization

! This file is part of MOM6. See LICENSE.md for the license.

use MOM_coms,          only : sum_across_PEs, PE_here
use MOM_error_handler, only : MOM_mesg, MOM_error, FATAL, WARNING, is_root_pe
use MOM_dyn_horgrid, only : dyn_horgrid_type
use MOM_file_parser, only : get_param, log_version, param_file_type
use MOM_get_input, only : directories
use MOM_grid, only : ocean_grid_type
use MOM_open_boundary, only : ocean_OBC_type, OBC_NONE, OBC_SIMPLE
use MOM_open_boundary, only : OBC_DIRECTION_E, OBC_DIRECTION_W, OBC_DIRECTION_N
use MOM_open_boundary, only : OBC_DIRECTION_S
use MOM_open_boundary,  only : OBC_segment_type, register_OBC
use MOM_open_boundary,  only : OBC_registry_type
use MOM_unit_scaling, only : unit_scale_type
use MOM_variables, only : thermo_var_ptrs
use MOM_verticalGrid, only : verticalGrid_type
implicit none ; private

#include <MOM_memory.h>


character(len=40) :: mdl = "jet_initialization" !< This module's name.

! The following routines are visible to the outside world
public jet_initialize_thickness, jet_initialize_velocity

! A note on unit descriptions in comments: MOM6 uses units that can be rescaled for dimensional
! consistency testing. These are noted in comments with units like Z, H, L, and T, along with
! their mks counterparts with notation like "a velocity [Z T-1 ~> m s-1]".  If the units
! vary with the Boussinesq approximation, the Boussinesq variant is given first.

!> A module variable that should not be used.
!! \todo Move this module variable into a control structure.
logical :: first_call = .true.
logical :: jet_CS_initialized = .false.

!> Control structure for jet properties.
type, public :: jet_CS_type ; private
  real :: f_0                   !! Coriolis parameter [T-1 ~> s-1]
  real :: delta                 !! asymmetry parameter
  real :: F                     !! Inverse Burger number
  real :: Ro                    !! Rossby number
  real :: L                     !! Length scale [L ~> km]
  real :: H1                    !! Upper layer depth at jet axis [L ~> m]
  real :: H                     !! Total depth [L ~> m]
  real :: U0                    !! Maximum velocity at jet axis [L S-1 ~> m s-1]
  real :: trans                 !! total transport [L^3 S-1 ~> m s-1]
end type jet_CS_type

type(jet_CS_type) jet_CS

contains

!> Get parameters
subroutine jet_get_parameters(param_file, G, GV)
  type(param_file_type),   intent(in)  :: param_file !< A structure indicating the open
                                             !! file to parse for model parameter values.
  type(ocean_grid_type),   intent(in) :: G  !< The ocean's grid structure.
  type(verticalGrid_type), intent(in)  :: GV !< The ocean's vertical grid structure.
  integer :: i, j, k, n, is, ie, js, je, isd, ied, jsd, jed, nz
  integer :: IsdB, IedB, JsdB, JedB
  real :: km2m = 1000.0
  real :: f_0, L, delta, Ro, F, H1, U0, trans, y
  character(len=256) :: mesg    ! Message for error messages.
    
  if (jet_CS_initialized) return
  jet_CS_initialized = .true.
    
  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; nz = GV%ke
  isd = G%isd ; ied = G%ied ; jsd = G%jsd ; jed = G%jed
  IsdB = G%IsdB ; IedB = G%IedB ; JsdB = G%JsdB ; JedB = G%JedB
  
    
  call get_param(param_file, "MOM_shared_initialization", "F_0", f_0, &
                 "The reference value of the Coriolis parameter with the "//&
                 "betaplane option.", units="s-1", default=0.0)
  call get_param(param_file, mdl,"JET_L", L, &
                 "Jet length scale.",&
                 units="km", default=31.0)
  call get_param(param_file, mdl, "JET_DELTA", delta, &
                 "Jet asymmetry parameter.",  &
                 units="nondim", default=0.3)
  call get_param(param_file, mdl, "JET_ROSSBY", Ro, &
                 "Jet Rossby number.",  &
                 units="nondim", default=0.6)
  call get_param(param_file, mdl, "JET_F", F, &
                 "Jet inverse Burger number.",  &
                 units="nondim", default=1.0989)
                 
  H1 = (f_0 * L*km2m)**2/(F * GV%g_prime(GV%ke))
  U0 = f_0 * L * km2m * Ro
  
  jet_CS%f_0   = f_0
  jet_CS%L     = L
  jet_CS%delta = delta
  jet_CS%Ro    = Ro
  jet_CS%F     = F
  jet_CS%H1    = H1
  jet_CS%U0    = U0
  
  print '("U0 = ", F8.4, "   H1 = ", F12.4)', &
        U0, H1
!   call MOM_mesg(trim(mesg), verb=4, all_print=.true.)

end subroutine jet_get_parameters

!> Calculate jet velocity
function jet_uvel(y, k)
  real, intent(in)         :: y         !! latitude (in km)
  integer, intent(in)      :: k         !! Vertical level
  
  real :: L, delta, U0
  real :: jet_uvel
  
  U0    = jet_CS%U0
  L     = jet_CS%L
  delta = jet_CS%delta
  
  if (k == 1) then
    if (y <= 0.0) then
      jet_uvel = U0*exp(y/(L*(1.0 + delta)))
    else
      jet_uvel = U0*exp(-y/(L*(1.0 - delta)))
    endif 
  else
    jet_uvel = 0.0
  endif

end function jet_uvel


!> Calculate jet thickness
function jet_thickness(y, k, G, GV)
  real,                    intent(in) :: y  !! latitude (in km)
  integer,                 intent(in) :: k  !! Vertical level
  type(ocean_grid_type),   intent(in) :: G  !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV !< The ocean's vertical grid structure.
  
  real :: L, delta, Ro, F, H1
  real :: jet_thickness
  
  H1    = jet_CS%H1
  F     = jet_CS%F
  Ro    = jet_CS%Ro
  delta = jet_CS%delta
  L     = jet_CS%L
  
 
  if (GV%ke == 1) then
    if (y <= 0.0) then
      jet_thickness = H1 + H1*F*Ro*(1.0 - (1.0 + delta)*exp(y/(L*(1.0 + delta))))
    else
      jet_thickness = H1 - H1*F*Ro*(1.0 - (1.0 - delta)*exp(-y/(L*(1.0 - delta))))
    endif
  else
    if (k == 1) then
      jet_thickness = H1 - (1.0 + GV%g_prime(2)/GV%g_prime(1)) * H1 * F * Ro &
                      * (sign(1.0,y) - delta) &
                      * (1.0 - exp(-abs(y)/(L * (1.0 - sign(delta, y)))))
    elseif (k == 2) then
      jet_thickness = G%max_depth - H1 + H1 * F * Ro &
                      * (sign(1.0,y) - delta) &
                      * (1.0 - exp(-abs(y)/(L * (1.0 - sign(delta, y)))))
    endif
  endif
end function jet_thickness

!> Calculate jet eta
function jet_eta(y, GV)
  real,                    intent(in) :: y  !! latitude (in km)
  type(verticalGrid_type), intent(in) :: GV !< The ocean's vertical grid structure.
  
  real :: L, delta, Ro, F, H1
  real :: jet_eta
  
  H1    = jet_CS%H1
  F     = jet_CS%F
  Ro    = jet_CS%Ro
  delta = jet_CS%delta
  L     = jet_CS%L
  
  jet_eta = - GV%g_prime(2)/GV%g_prime(1) * H1 * F * Ro &
                    * (sign(1.0,y) - delta) &
                    * (1.0 - exp(-abs(y)/(L * (1.0 - sign(delta, y)))))
end function jet_eta



!> initialize thicknesses.
subroutine jet_initialize_thickness(h, G, GV, param_file, just_read)
  type(ocean_grid_type),   intent(in)  :: G  !< The ocean's grid structure.
  type(verticalGrid_type), intent(in)  :: GV !< The ocean's vertical grid structure.
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), &
                           intent(out) :: h  !< The thicknesses being initialized [H ~> m or kg m-2].
  type(param_file_type),   intent(in)  :: param_file !< A structure indicating the open
                                             !! file to parse for model parameter values.
  logical,                 intent(in)  :: just_read !< If true, this call will
                                             !! only read parameters without changing h.

  real    :: y
  integer :: i, j, k, is, ie, js, je, nz

  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; nz = GV%ke
  
  if (just_read) return ! All run-time parameters have been read, so return.

  call jet_get_parameters(param_file, G, GV)

  h(:,:,:) = 0

  do k=1,nz
    do j=js,je ; do i=is,ie
      y = G%geoLatT(I,j)
    
      h(i,j,k) = jet_thickness(y, k, G, GV)
    enddo ; enddo
  enddo
  
  if (first_call) call write_user_log(param_file)

end subroutine jet_initialize_thickness


!> initialize velocities.
subroutine jet_initialize_velocity(u, v, G, GV, US, param_file, just_read)
  type(ocean_grid_type),                       intent(in)  :: G !< Ocean grid structure.
  type(verticalGrid_type),                     intent(in)  :: GV !< The ocean's vertical grid structure.
  real, dimension(SZIB_(G), SZJ_(G),SZK_(GV)), intent(out) :: u !< i-component of velocity [L T-1 ~> m s-1]
  real, dimension(SZI_(G), SZJB_(G),SZK_(GV)), intent(out) :: v !< j-component of velocity [L T-1 ~> m s-1]
  type(unit_scale_type),                       intent(in)  :: US !< A dimensional unit scaling type
  type(param_file_type),                       intent(in)  :: param_file !< A structure indicating the
                                                            !! open file to parse for model
                                                            !! parameter values.
  logical,                                     intent(in)  :: just_read !< If true, this call will
                                                      !! only read parameters without changing u & v.

  integer :: i, j, k, n, is, ie, js, je, nz
  real    :: y

  if (just_read) return ! All run-time parameters have been read, so return.

  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; nz = GV%ke

  u(:,:,:) = 0.0
  v(:,:,:) = 0.0

  if (first_call) call write_user_log(param_file)

  call jet_get_parameters(param_file, G, GV)

  do j = js,je ; do I = is-1,ie+1
    y = G%geoLatCu(I,j)
    u(I,j,1) = jet_uvel(y, 1)
  enddo ; enddo
  
end subroutine jet_initialize_velocity


!> Write output about the parameter values being used.
subroutine write_user_log(param_file)
  type(param_file_type), intent(in) :: param_file !< A structure indicating the
                                                  !! open file to parse for model
                                                  !! parameter values.

  ! This include declares and sets the variable "version".
# include "version_variable.h"
  character(len=40)  :: mdl = "jet_initialization" ! This module's name.

  call log_version(param_file, mdl, version)
  first_call = .false.

end subroutine write_user_log

!> \namespace user_initialization
!!
!!  This subroutine initializes the fields for the simulations.
!!  The one argument passed to initialize, Time, is set to the
!!  current time of the simulation.  The fields which might be initialized
!!  here are:
!!  - u - Zonal velocity [Z T-1 ~> m s-1].
!!  - v - Meridional velocity [Z T-1 ~> m s-1].
!!  - h - Layer thickness [H ~> m or kg m-2].  (Must be positive.)
!!  - G%bathyT - Basin depth [Z ~> m].
!!  - G%CoriolisBu - The Coriolis parameter [T-1 ~> s-1].
!!  - GV%g_prime - The reduced gravity at each interface [L2 Z-1 T-2 ~> m s-2].
!!  - GV%Rlay - Layer potential density (coordinate variable) [R ~> kg m-3].
!!  If ENABLE_THERMODYNAMICS is defined:
!!  - T - Temperature [degC].
!!  - S - Salinity [ppt].
!!  If BULKMIXEDLAYER is defined:
!!  - Rml - Mixed layer and buffer layer potential densities [R ~> kg m-3].
!!  If SPONGE is defined:
!!  - A series of subroutine calls are made to set up the damping
!!    rates and reference profiles for all variables that are damped
!!    in the sponge.
!!
!!  Any user provided tracer code is also first linked through this
!!  subroutine.
!!
!!  These variables are all set in the set of subroutines (in this
!!  file) jet_initialize_bottom_depth, jet_initialize_thickness,
!!  jet_initialize_velocity,  jet_initialize_temperature_salinity,
!!  jet_initialize_mixed_layer_density, jet_initialize_sponges,
!!  jet_set_coord, and jet_set_ref_profile.
!!
!!  The names of these subroutines should be self-explanatory. They
!!  start with "jet_" to indicate that they will likely have to be
!!  modified for each simulation to set the initial conditions and
!!  boundary conditions.  Most of these take two arguments: an integer
!!  argument specifying whether the fields are to be calculated
!!  internally or read from a NetCDF file; and a string giving the
!!  path to that file.  If the field is initialized internally, the
!!  path is ignored.

end module jet_initialization
