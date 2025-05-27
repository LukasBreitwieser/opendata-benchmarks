// @(#)root/mathcore:$Id: 04c6d98020d7178ed5f0884f9466bca32b031565 $
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
*                                                                    *
* Copyright (c) 2005 , LCG ROOT MathLib Team                         *
*                                                                    *
*                                                                    *
**********************************************************************/

// Header file for class DevicePxPyPzE4D
//
// Created by: fischler at Wed Jul 20   2005
//   (starting from DevicePxPyPzE4D by moneta)
//
// Last update: $Id: 04c6d98020d7178ed5f0884f9466bca32b031565 $
//
#ifndef DEVICE_PX_PY_PZ_E_4D_H_
#define DEVICE_PX_PY_PZ_E_4D_H_

//#include "Math/GenVector/eta.h"

//#include "Math/GenVector/GenVector_exception.h"


//#include <__clang_cuda_runtime_wrapper.h>
#include <cmath>
#include <stdexcept>


//__________________________________________________________________________________________
/**
    Class describing a 4D cartesian coordinate system (x, y, z, t coordinates)
    or momentum-energy vectors stored as (Px, Py, Pz, E).
    The metric used is (-,-,-,+)

    @ingroup GenVector

    @sa Overview of the @ref GenVector "physics vector library"
*/

template <class ScalarType = double>
class DevicePxPyPzE4D {

public :

   typedef ScalarType Scalar;
   static constexpr unsigned int Dimension = 4U;

   // --------- Constructors ---------------

   /**
      Default constructor  with x=y=z=t=0
   */
   __device__ DevicePxPyPzE4D() : fX(0.0), fY(0.0), fZ(0.0), fT(0.0) { }


   /**
      Constructor  from x, y , z , t values
   */
   __device__ DevicePxPyPzE4D(Scalar px, Scalar py, Scalar pz, Scalar e) :
      fX(px), fY(py), fZ(pz), fT(e) { }


   /**
      construct from any vector or  coordinate system class
      implementing x(), y() and z() and t()
   */
   template <class CoordSystem>
   explicit constexpr DevicePxPyPzE4D(const CoordSystem & v) :
      fX( v.x() ), fY( v.y() ), fZ( v.z() ), fT( v.t() )  { }

   // for g++  3.2 and 3.4 on 32 bits found that the compiler generated copy ctor and assignment are much slower
   // so we decided to re-implement them ( there is no no need to have them with g++4)
   /**
      copy constructor
    */
   __device__ DevicePxPyPzE4D(const DevicePxPyPzE4D & v) :
      fX(v.fX), fY(v.fY), fZ(v.fZ), fT(v.fT) { }

   /**
      assignment operator
    */
   __device__ DevicePxPyPzE4D & operator = (const DevicePxPyPzE4D & v) {
      fX = v.fX;
      fY = v.fY;
      fZ = v.fZ;
      fT = v.fT;
      return *this;
   }

   /**
      Set internal data based on an array of 4 Scalar numbers
   */
   void SetCoordinates( const Scalar src[] )
   { fX=src[0]; fY=src[1]; fZ=src[2]; fT=src[3]; }

   /**
      get internal data into an array of 4 Scalar numbers
   */
   void GetCoordinates( Scalar dest[] ) const
   { dest[0] = fX; dest[1] = fY; dest[2] = fZ; dest[3] = fT; }

   /**
      Set internal data based on 4 Scalar numbers
   */
   void SetCoordinates(Scalar  px, Scalar  py, Scalar  pz, Scalar e)
   { fX=px; fY=py; fZ=pz; fT=e;}

   /**
      get internal data into 4 Scalar numbers
   */
   void GetCoordinates(Scalar& px, Scalar& py, Scalar& pz, Scalar& e) const
   { px=fX; py=fY; pz=fZ; e=fT;}

   // --------- Coordinates and Coordinate-like Scalar properties -------------

   // cartesian (Minkowski)coordinate accessors

   Scalar Px() const { return fX;}
   Scalar Py() const { return fY;}
   Scalar Pz() const { return fZ;}
   Scalar E()  const { return fT;}

   Scalar X() const { return fX;}
   Scalar Y() const { return fY;}
   Scalar Z() const { return fZ;}
   Scalar T() const { return fT;}

   // other coordinate representation

   /**
      squared magnitude of spatial components
   */
   Scalar P2() const { return fX*fX + fY*fY + fZ*fZ; }

   /**
      magnitude of spatial components (magnitude of 3-momentum)
   */
   Scalar P() const { using std::sqrt; return sqrt(P2()); }
   Scalar R() const { return P(); }

   /**
      vector magnitude squared (or mass squared)
   */
   Scalar M2() const   { return fT*fT - fX*fX - fY*fY - fZ*fZ;}
   Scalar Mag2() const { return M2(); }

   /**
      invariant mass
   */
   Scalar M() const
   {
      const Scalar mm = M2();
      if (mm >= 0) {
         using std::sqrt;
         return sqrt(mm);
      } else {
        throw std::runtime_error ("DevicePxPyPzE4D::M() - Tachyonic:\n"
                   "    P^2 > E^2 so the mass would be imaginary");
         using std::sqrt;
         return -sqrt(-mm);
      }
   }
   Scalar Mag() const    { return M(); }

   /**
       transverse spatial component squared
   */
   Scalar Pt2()   const { return fX*fX + fY*fY;}
   Scalar Perp2() const { return Pt2();}

   /**
      Transverse spatial component (P_perp or rho)
   */
   Scalar Pt() const { using std::sqrt; return sqrt(Perp2()); }
   Scalar Perp() const { return Pt();}
   Scalar Rho()  const { return Pt();}

   /**
       transverse mass squared
   */
   Scalar Mt2() const { return fT*fT - fZ*fZ; }

   /**
      transverse mass
   */
   Scalar Mt() const {
      const Scalar mm = Mt2();
      if (mm >= 0) {
         using std::sqrt;
         return sqrt(mm);
      } else {
         throw std::runtime_error("DevicePxPyPzE4D::Mt() - Tachyonic:\n"
                           "    Pz^2 > E^2 so the transverse mass would be imaginary");
         using std::sqrt;
         return -sqrt(-mm);
      }
   }

   /**
       transverse energy squared
   */
   Scalar Et2() const {  // is (E^2 * pt ^2) / p^2
      // but it is faster to form p^2 from pt^2
      Scalar pt2 = Pt2();
      return pt2 == 0 ? 0 : fT*fT * pt2/( pt2 + fZ*fZ );
   }

   /**
      transverse energy
   */
   Scalar Et() const {
      const Scalar etet = Et2();
      using std::sqrt;
      return fT < 0.0 ? -sqrt(etet) : sqrt(etet);
   }

   /**
      azimuthal angle
   */
   Scalar Phi() const { using std::atan2; return (fX == 0.0 && fY == 0.0) ? 0 : atan2(fY, fX); }

   /**
      polar angle
   */
   Scalar Theta() const { using std::atan2; return (fX == 0.0 && fY == 0.0 && fZ == 0.0) ? 0 : atan2(Pt(), fZ); }

   /**
       pseudorapidity
   */
   //Scalar Eta() const {
      //return Impl::Eta_FromRhoZ ( Pt(), fZ);
   //}

   // --------- Set Coordinates of this system  ---------------


   /**
      set X value
   */
   void SetPx( Scalar  px) {
      fX = px;
   }
   /**
      set Y value
   */
   void SetPy( Scalar  py) {
      fY = py;
   }
   /**
      set Z value
   */
   void SetPz( Scalar  pz) {
      fZ = pz;
   }
   /**
      set T value
   */
   void SetE( Scalar  e) {
      fT = e;
   }

   /**
       set all values using cartesian coordinates
   */
   void SetPxPyPzE(Scalar px, Scalar py, Scalar pz, Scalar e) {
      fX=px;
      fY=py;
      fZ=pz;
      fT=e;
   }



   // ------ Manipulations -------------

   /**
      negate the 4-vector
   */
   void Negate( ) { fX = -fX; fY = -fY;  fZ = -fZ; fT = -fT;}

   /**
      scale coordinate values by a scalar quantity a
   */
   void Scale( const Scalar & a) {
      fX *= a;
      fY *= a;
      fZ *= a;
      fT *= a;
   }

   /**
      Assignment from a generic coordinate system implementing
      x(), y(), z() and t()
   */
   template <class AnyCoordSystem>
   DevicePxPyPzE4D & operator = (const AnyCoordSystem & v) {
      fX = v.x();
      fY = v.y();
      fZ = v.z();
      fT = v.t();
      return *this;
   }

   /**
      Exact equality
   */
   bool operator == (const DevicePxPyPzE4D & rhs) const {
      return fX == rhs.fX && fY == rhs.fY && fZ == rhs.fZ && fT == rhs.fT;
   }
   bool operator != (const DevicePxPyPzE4D & rhs) const {return !(operator==(rhs));}


   // ============= Compatibility section ==================

   // The following make this coordinate system look enough like a CLHEP
   // vector that an assignment member template can work with either
   Scalar x() const { return fX; }
   Scalar y() const { return fY; }
   Scalar z() const { return fZ; }
   Scalar t() const { return fT; }



#if defined(__MAKECINT__) || defined(G__DICTIONARY)

   // ====== Set member functions for coordinates in other systems =======

   void SetPt(Scalar pt);

   void SetEta(Scalar eta);

   void SetPhi(Scalar phi);

   void SetM(Scalar m);

#endif

private:

   /**
      (contiguous) data containing the coordinate values x,y,z,t
   */

   ScalarType fX;
   ScalarType fY;
   ScalarType fZ;
   ScalarType fT;

};


#endif // DEVICE_PX_PY_PZ_E_4D_H_
