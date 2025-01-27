PROGRAM test
  ! Parameter list
  IMPLICIT NONE
  REAL :: E, nu, sigy0, H, cid, g, davg

  ! Assign parameter values
  E = 1.0
  nu = 1.0
  sigy0 = 1.0
  H = 1.0
  cid = 1.0

  ! Compute trial stress and some other values
  g = 0.5 * E / (1.0 + nu)
  davg = H * 2.0

  ! Print results
  PRINT *, "Hello, world!"
  PRINT *, "g = ", g
  PRINT *, "davg = ", davg

END PROGRAM test
