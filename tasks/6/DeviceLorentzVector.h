// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class DeviceLorentzVector
//
// Created by:    moneta   at Tue May 31 17:06:09 2005
// Major mods by: fischler at Wed Jul 20   2005
//
// Last update: $Id$
//
#ifndef DEVICELORENTZVECTOR_H_
#define DEVICELORENTZVECTOR_H_

#include "DevicePxPyPzE4D.h"

//#include "Math/GenVector/DisplacementVector3D.h"

//#include "Math/GenVector/GenVectorIO.h"

#include <cmath>
#include <string>
#include <stdexcept>


//__________________________________________________________________________________________
/** @ingroup GenVector

Class describing a generic DeviceLorentzVector in the 4D space-time,
using the specified coordinate system for the spatial vector part.
The metric used for the DeviceLorentzVector is (-,-,-,+).
In the case of DeviceLorentzVector we don't distinguish the concepts
of points and displacement vectors as in the 3D case,
since the main use case for 4D Vectors is to describe the kinematics of
relativistic particles. A DeviceLorentzVector behaves like a
DisplacementVector in 4D.  The Minkowski components could be viewed as
v and t, or for kinematic 4-vectors, as p and E.

ROOT provides specialisations and aliases to them of the ROOT::Math::DeviceLorentzVector template:
- ROOT::Math::PtEtaPhiMVector based on pt (rho),eta,phi and M (t) coordinates in double precision
- ROOT::Math::PtEtaPhiEVector based on pt (rho),eta,phi and E (t) coordinates in double precision
- ROOT::Math::PxPyPzMVector based on px,py,pz and M (mass) coordinates in double precision
- ROOT::Math::PxPyPzEVector based on px,py,pz and E (energy) coordinates in double precision
- ROOT::Math::XYZTVector based on x,y,z,t coordinates (cartesian) in double precision (same as PxPyPzEVector)
- ROOT::Math::XYZTVectorF based on x,y,z,t coordinates (cartesian) in float precision (same as PxPyPzEVector but float)

@sa Overview of the @ref GenVector "physics vector library"
*/

    template< class CoordSystem >
    class DeviceLorentzVector {

    public:

       // ------ ctors ------

       typedef typename CoordSystem::Scalar Scalar;
       typedef CoordSystem CoordinateType;

       /**
          default constructor of an empty vector (Px = Py = Pz = E = 0 )
       */
       __device__ DeviceLorentzVector ( ) : fCoordinates() { }

       /**
          generic constructors from four scalar values.
          The association between values and coordinate depends on the
          coordinate system.  For DevicePxPyPzE4D,
          \param a scalar value (Px)
          \param b scalar value (Py)
          \param c scalar value (Pz)
          \param d scalar value (E)
       */
       __device__ DeviceLorentzVector(const Scalar & a,
                     const Scalar & b,
                     const Scalar & c,
                     const Scalar & d) :
          fCoordinates(a , b,  c, d)  { }

       /**
          constructor from a DeviceLorentzVector expressed in different
          coordinates, or using a different Scalar type
       */
       template< class Coords >
       __device__ explicit constexpr DeviceLorentzVector(const DeviceLorentzVector<Coords> & v ) :
          fCoordinates( v.Coordinates() ) { }

       /**
          Construct from a foreign 4D vector type, for example, HepDeviceLorentzVector
          Precondition: v must implement methods x(), y(), z(), and t()
       */
       template<class ForeignDeviceLorentzVector,
                typename = decltype(std::declval<ForeignDeviceLorentzVector>().x()
                                    + std::declval<ForeignDeviceLorentzVector>().y()
                                    + std::declval<ForeignDeviceLorentzVector>().z()
                                    + std::declval<ForeignDeviceLorentzVector>().t())>
       __device__ explicit constexpr DeviceLorentzVector( const ForeignDeviceLorentzVector & v) :
          fCoordinates(DevicePxPyPzE4D<Scalar>( v.x(), v.y(), v.z(), v.t()  ) ) { }

#ifdef LATER
       /**
          construct from a generic linear algebra  vector implementing operator []
          and with a size of at least 4. This could be also a C array
          In this case v[0] is the first data member
          ( Px for a DevicePxPyPzE4D base)
          \param v LA vector
          \param index0 index of first vector element (Px)
       */
       template< class LAVector >
       explicit constexpr DeviceLorentzVector(const LAVector & v, size_t index0 ) {
          fCoordinates = CoordSystem ( v[index0], v[index0+1], v[index0+2], v[index0+3] );
       }
#endif


       // ------ assignment ------

       /**
          Assignment operator from a lorentz vector of arbitrary type
       */
       template< class OtherCoords >
       DeviceLorentzVector & operator= ( const DeviceLorentzVector<OtherCoords> & v) {
          fCoordinates = v.Coordinates();
          return *this;
       }

       /**
          assignment from any other Lorentz vector  implementing
          x(), y(), z() and t()
       */
       template<class ForeignDeviceLorentzVector,
                typename = decltype(std::declval<ForeignDeviceLorentzVector>().x()
                                    + std::declval<ForeignDeviceLorentzVector>().y()
                                    + std::declval<ForeignDeviceLorentzVector>().z()
                                    + std::declval<ForeignDeviceLorentzVector>().t())>
       DeviceLorentzVector & operator = ( const ForeignDeviceLorentzVector & v) {
          SetXYZT( v.x(), v.y(), v.z(), v.t() );
          return *this;
       }

#ifdef LATER
       /**
          assign from a generic linear algebra  vector implementing operator []
          and with a size of at least 4
          In this case v[0] is the first data member
          ( Px for a DevicePxPyPzE4D base)
          \param v LA vector
          \param index0 index of first vector element (Px)
       */
       template< class LAVector >
       DeviceLorentzVector & AssignFrom(const LAVector & v, size_t index0=0 ) {
          fCoordinates.SetCoordinates( v[index0], v[index0+1], v[index0+2], v[index0+3] );
          return *this;
       }
#endif

       // ------ Set, Get, and access coordinate data ------

       /**
          Retrieve a const reference to  the coordinates object
       */
       const CoordSystem & Coordinates() const {
          return fCoordinates;
       }

       /**
          Set internal data based on an array of 4 Scalar numbers
       */
       DeviceLorentzVector<CoordSystem>& SetCoordinates( const Scalar src[] ) {
          fCoordinates.SetCoordinates(src);
          return *this;
       }

       /**
          Set internal data based on 4 Scalar numbers
       */
       DeviceLorentzVector<CoordSystem>& SetCoordinates( Scalar a, Scalar b, Scalar c, Scalar d ) {
          fCoordinates.SetCoordinates(a, b, c, d);
          return *this;
       }

       /**
          Set internal data based on 4 Scalars at *begin to *end
       */
       template< class IT >
       DeviceLorentzVector<CoordSystem>& SetCoordinates( IT begin, IT end  ) {
          IT a = begin; IT b = ++begin; IT c = ++begin; IT d = ++begin;
          (void)end;
          assert (++begin==end);
          SetCoordinates (*a,*b,*c,*d);
          return *this;
       }

       /**
          get internal data into 4 Scalar numbers
       */
       void GetCoordinates( Scalar& a, Scalar& b, Scalar& c, Scalar & d ) const
       { fCoordinates.GetCoordinates(a, b, c, d);  }

       /**
          get internal data into an array of 4 Scalar numbers
       */
       void GetCoordinates( Scalar dest[] ) const
       { fCoordinates.GetCoordinates(dest);  }

       /**
          get internal data into 4 Scalars at *begin to *end
       */
       template <class IT>
       void GetCoordinates( IT begin, IT end ) const
       { IT a = begin; IT b = ++begin; IT c = ++begin; IT d = ++begin;
       (void)end;
       assert (++begin==end);
       GetCoordinates (*a,*b,*c,*d);
       }

       /**
          get internal data into 4 Scalars at *begin
       */
       template <class IT>
       void GetCoordinates( IT begin ) const {
          Scalar a,b,c,d = 0;
          GetCoordinates (a,b,c,d);
          *begin++ = a;
          *begin++ = b;
          *begin++ = c;
          *begin   = d;
       }

       /**
          set the values of the vector from the cartesian components (x,y,z,t)
          (if the vector is held in another coordinates, like (Pt,eta,phi,m)
          then (x, y, z, t) are converted to that form)
       */
       __device__ DeviceLorentzVector<CoordSystem>& SetXYZT (Scalar xx, Scalar yy, Scalar zz, Scalar tt) {
          fCoordinates.SetPxPyPzE(xx,yy,zz,tt);
          return *this;
       }
       __device__ DeviceLorentzVector<CoordSystem>& SetPxPyPzE (Scalar xx, Scalar yy, Scalar zz, Scalar ee) {
          fCoordinates.SetPxPyPzE(xx,yy,zz,ee);
          return *this;
       }

       // ------------------- Equality -----------------

       /**
          Exact equality
       */
       bool operator==(const DeviceLorentzVector & rhs) const {
          return fCoordinates==rhs.fCoordinates;
       }
       bool operator!= (const DeviceLorentzVector & rhs) const {
          return !(operator==(rhs));
       }

       // ------ Individual element access, in various coordinate systems ------

       /**
          dimension
       */
       unsigned int Dimension() const
       {
          return fDimension;
       };

       // individual coordinate accessors in various coordinate systems

       /**
          spatial X component
       */
       Scalar Px() const  { return fCoordinates.Px(); }
       __device__ Scalar X()  const  { return fCoordinates.Px(); }
       /**
          spatial Y component
       */
       Scalar Py() const { return fCoordinates.Py(); }
       Scalar Y()  const { return fCoordinates.Py(); }
       /**
          spatial Z component
       */
       Scalar Pz() const { return fCoordinates.Pz(); }
       Scalar Z()  const { return fCoordinates.Pz(); }
       /**
          return 4-th component (time, or energy for a 4-momentum vector)
       */
       __device__ Scalar E()  const { return fCoordinates.E(); }
       Scalar T()  const { return fCoordinates.E(); }
       /**
          return magnitude (mass) squared  M2 = T**2 - X**2 - Y**2 - Z**2
          (we use -,-,-,+ metric)
       */
       Scalar M2()   const { return fCoordinates.M2(); }
       /**
          return magnitude (mass) using the  (-,-,-,+)  metric.
          If M2 is negative (space-like vector) a GenVector_exception
          is suggested and if continuing, - sqrt( -M2) is returned
       */
       Scalar M() const    { return fCoordinates.M();}
       /**
          return the spatial (3D) magnitude ( sqrt(X**2 + Y**2 + Z**2) )
       */
       Scalar R() const { return fCoordinates.R(); }
       Scalar P() const { return fCoordinates.R(); }
       /**
          return the square of the spatial (3D) magnitude ( X**2 + Y**2 + Z**2 )
       */
       Scalar P2() const { return P() * P(); }
       /**
          return the square of the transverse spatial component ( X**2 + Y**2 )
       */
       Scalar Perp2( ) const { return fCoordinates.Perp2();}

       /**
          return the  transverse spatial component sqrt ( X**2 + Y**2 )
       */
       Scalar Pt()  const { return fCoordinates.Pt(); }
       Scalar Rho() const { return fCoordinates.Pt(); }

       /**
          return the transverse mass squared
          \f[ m_t^2 = E^2 - p{_z}^2 \f]
       */
       Scalar Mt2() const { return fCoordinates.Mt2(); }

       /**
          return the transverse mass
          \f[ \sqrt{ m_t^2 = E^2 - p{_z}^2} X sign(E^ - p{_z}^2) \f]
       */
       Scalar Mt() const { return fCoordinates.Mt(); }

       /**
          return the transverse energy squared
          \f[ e_t = \frac{E^2 p_{\perp}^2 }{ |p|^2 } \f]
       */
       Scalar Et2() const { return fCoordinates.Et2(); }

       /**
          return the transverse energy
          \f[ e_t = \sqrt{ \frac{E^2 p_{\perp}^2 }{ |p|^2 } } X sign(E) \f]
       */
       Scalar Et() const { return fCoordinates.Et(); }

       /**
          azimuthal  Angle
       */
       Scalar Phi() const  { return fCoordinates.Phi();}

       /**
          polar Angle
       */
       Scalar Theta() const { return fCoordinates.Theta(); }

       /**
          pseudorapidity
          \f[ \eta = - \ln { \tan { \frac { \theta} {2} } } \f]
       */
       Scalar Eta() const { return fCoordinates.Eta(); }

       /**
          get the spatial components of the Vector in a
          DisplacementVector based on Cartesian Coordinates
       */
       //::ROOT::Math::DisplacementVector3D<Cartesian3D<Scalar> > Vect() const {
          //return ::ROOT::Math::DisplacementVector3D<Cartesian3D<Scalar> >( X(), Y(), Z() );
       //}

       // ------ Operations combining two Lorentz vectors ------

       /**
          scalar (Dot) product of two DeviceLorentzVector vectors (metric is -,-,-,+)
          Enable the product using any other DeviceLorentzVector implementing
          the x(), y() , y() and t() member functions
          \param  q  any DeviceLorentzVector implementing the x(), y() , z() and t()
          member functions
          \return the result of v.q of type according to the base scalar type of v
       */

       template< class OtherDeviceLorentzVector >
       Scalar Dot(const OtherDeviceLorentzVector & q) const {
          return t()*q.t() - x()*q.x() - y()*q.y() - z()*q.z();
       }

       /**
          Self addition with another Vector ( v+= q )
          Enable the addition with any other DeviceLorentzVector
          \param  q  any DeviceLorentzVector implementing the x(), y() , z() and t()
          member functions
       */
      template< class OtherDeviceLorentzVector >
      __device__ inline DeviceLorentzVector & operator += ( const OtherDeviceLorentzVector & q)
       {
          SetXYZT( x() + q.x(), y() + q.y(), z() + q.z(), t() + q.t()  );
          return *this;
       }

       /**
          Self subtraction of another Vector from this ( v-= q )
          Enable the addition with any other DeviceLorentzVector
          \param  q  any DeviceLorentzVector implementing the x(), y() , z() and t()
          member functions
       */
       template< class OtherDeviceLorentzVector >
       DeviceLorentzVector & operator -= ( const OtherDeviceLorentzVector & q) {
          SetXYZT( x() - q.x(), y() - q.y(), z() - q.z(), t() - q.t()  );
          return *this;
       }

       /**
          addition of two DeviceLorentzVectors (v3 = v1 + v2)
          Enable the addition with any other DeviceLorentzVector
          \param  v2  any DeviceLorentzVector implementing the x(), y() , z() and t()
          member functions
          \return a new DeviceLorentzVector of the same type as v1
       */
       template<class OtherDeviceLorentzVector>
       __device__ DeviceLorentzVector  operator +  ( const OtherDeviceLorentzVector & v2) const
       {
          DeviceLorentzVector<CoordinateType> v3(*this);
          v3 += v2;
          return v3;
       }

       /**
          subtraction of two DeviceLorentzVectors (v3 = v1 - v2)
          Enable the subtraction of any other DeviceLorentzVector
          \param  v2  any DeviceLorentzVector implementing the x(), y() , z() and t()
          member functions
          \return a new DeviceLorentzVector of the same type as v1
       */
       template<class OtherDeviceLorentzVector>
       DeviceLorentzVector  operator -  ( const OtherDeviceLorentzVector & v2) const {
          DeviceLorentzVector<CoordinateType> v3(*this);
          v3 -= v2;
          return v3;
       }

       //--- scaling operations ------

       /**
          multiplication by a scalar quantity v *= a
       */
       DeviceLorentzVector & operator *= (Scalar a) {
          fCoordinates.Scale(a);
          return *this;
       }

       /**
          division by a scalar quantity v /= a
       */
       DeviceLorentzVector & operator /= (Scalar a) {
          fCoordinates.Scale(1/a);
          return *this;
       }

       /**
          product of a DeviceLorentzVector by a scalar quantity
          \param a  scalar quantity of type a
          \return a new mathcoreDeviceLorentzVector q = v * a same type as v
       */
       DeviceLorentzVector operator * ( const Scalar & a) const {
          DeviceLorentzVector tmp(*this);
          tmp *= a;
          return tmp;
       }

       /**
          Divide a DeviceLorentzVector by a scalar quantity
          \param a  scalar quantity of type a
          \return a new mathcoreDeviceLorentzVector q = v / a same type as v
       */
       DeviceLorentzVector<CoordSystem> operator / ( const Scalar & a) const {
          DeviceLorentzVector<CoordSystem> tmp(*this);
          tmp /= a;
          return tmp;
       }

       /**
          Negative of a DeviceLorentzVector (q = - v )
          \return a new DeviceLorentzVector with opposite direction and time
       */
       DeviceLorentzVector operator - () const {
          //DeviceLorentzVector<CoordinateType> v(*this);
          //v.Negate();
          return operator*( Scalar(-1) );
       }
       DeviceLorentzVector operator + () const {
          return *this;
       }

       // ---- Relativistic Properties ----

       /**
          Rapidity relative to the Z axis:  .5 log [(E+Pz)/(E-Pz)]
       */
       Scalar Rapidity() const {
          // TODO - It would be good to check that E > Pz and use the Throw()
          //        mechanism or at least load a NAN if not.
          //        We should then move the code to a .cpp file.
          const Scalar ee  = E();
          const Scalar ppz = Pz();
          using std::log;
          return Scalar(0.5) * log((ee + ppz) / (ee - ppz));
       }

       /**
          Rapidity in the direction of travel: atanh (|P|/E)=.5 log[(E+P)/(E-P)]
       */
       Scalar ColinearRapidity() const {
          // TODO - It would be good to check that E > P and use the Throw()
          //        mechanism or at least load a NAN if not.
          const Scalar ee = E();
          const Scalar pp = P();
          using std::log;
          return Scalar(0.5) * log((ee + pp) / (ee - pp));
       }

       /**
          Determine if momentum-energy can represent a physical massive particle
       */
       bool isTimelike( ) const {
          Scalar ee = E(); Scalar pp = P(); return ee*ee > pp*pp;
       }

       /**
          Determine if momentum-energy can represent a massless particle
       */
       bool isLightlike( Scalar tolerance
                         = 100*std::numeric_limits<Scalar>::epsilon() ) const {
          Scalar ee = E(); Scalar pp = P(); Scalar delta = ee-pp;
          if ( ee==0 ) return pp==0;
          return delta*delta < tolerance * ee*ee;
       }

       /**
          Determine if momentum-energy is spacelike, and represents a tachyon
       */
       bool isSpacelike( ) const {
          Scalar ee = E(); Scalar pp = P(); return ee*ee < pp*pp;
       }

       //typedef DisplacementVector3D< Cartesian3D<Scalar> > BetaVector;

       /**
          The beta vector for the boost that would bring this vector into
          its center of mass frame (zero momentum)
       */
       //BetaVector BoostToCM( ) const {
          //if (E() == 0) {
             //if (P() == 0) {
                //return BetaVector();
             //} else {
                //// TODO - should attempt to Throw with msg about
                //// boostVector computed for DeviceLorentzVector with t=0
                //return -Vect()/E();
             //}
          //}
          //if (M2() <= 0) {
             //// TODO - should attempt to Throw with msg about
             //// boostVector computed for a non-timelike DeviceLorentzVector
          //}
          //return -Vect()/E();
       //}

       //[>*
          //The beta vector for the boost that would bring this vector into
          //its center of mass frame (zero momentum)
       //*/
       //template <class Other4Vector>
       //BetaVector BoostToCM(const Other4Vector& v ) const {
          //Scalar eSum = E() + v.E();
          //DisplacementVector3D< Cartesian3D<Scalar> > vecSum = Vect() + v.Vect();
          //if (eSum == 0) {
             //if (vecSum.Mag2() == 0) {
                //return BetaVector();
             //} else {
                //// TODO - should attempt to Throw with msg about
                //// boostToCM computed for two 4-vectors with combined t=0
                //return BetaVector(vecSum/eSum);
             //}
             //// TODO - should attempt to Throw with msg about
             //// boostToCM computed for two 4-vectors with combined e=0
          //}
          //return BetaVector (vecSum * (-1./eSum));
       //}

       //beta and gamma

       /**
           Return beta scalar value
       */
       Scalar Beta() const {
          if ( E() == 0 ) {
             if ( P2() == 0)
                // to avoid Nan
                return 0;
             else {
                throw std::runtime_error ("DeviceLorentzVector::Beta() - beta computed for DeviceLorentzVector with t = 0. Return an Infinite result");
                return 1./E();
             }
          }
          if ( M2() <= 0 ) {
             throw std::runtime_error ("DeviceLorentzVector::Beta() - beta computed for non-timelike DeviceLorentzVector . Result is physically meaningless" );
          }
          return P() / E();
       }
       /**
           Return Gamma scalar value
       */
       Scalar Gamma() const {
          const Scalar v2 = P2();
          const Scalar t2 = E() * E();
          if (E() == 0) {
             if ( P2() == 0) {
                return 1;
             } else {
                throw std::runtime_error ("DeviceLorentzVector::Gamma() - gamma computed for DeviceLorentzVector with t = 0. Return a zero result");

             }
          }
          if ( t2 < v2 ) {
             throw std::runtime_error ("DeviceLorentzVector::Gamma() - gamma computed for a spacelike DeviceLorentzVector. Imaginary result");
             return 0;
          }
          else if ( t2 == v2 ) {
             throw std::runtime_error ("DeviceLorentzVector::Gamma() - gamma computed for a lightlike DeviceLorentzVector. Infinite result");
          }
          using std::sqrt;
          return Scalar(1) / sqrt(Scalar(1) - v2 / t2);
       } /* gamma */


       // Method providing limited backward name compatibility with CLHEP ----

       __device__ Scalar x()     const { return fCoordinates.Px();     }
       __device__ Scalar y()     const { return fCoordinates.Py();     }
       __device__ Scalar z()     const { return fCoordinates.Pz();     }
       __device__ Scalar t()     const { return fCoordinates.E();      }
       Scalar px()    const { return fCoordinates.Px();     }
       Scalar py()    const { return fCoordinates.Py();     }
       Scalar pz()    const { return fCoordinates.Pz();     }
       Scalar e()     const { return fCoordinates.E();      }
       Scalar r()     const { return fCoordinates.R();      }
       Scalar theta() const { return fCoordinates.Theta();  }
       Scalar phi()   const { return fCoordinates.Phi();    }
       Scalar rho()   const { return fCoordinates.Rho();    }
       Scalar eta()   const { return fCoordinates.Eta();    }
       Scalar pt()    const { return fCoordinates.Pt();     }
       Scalar perp2() const { return fCoordinates.Perp2();  }
       Scalar mag2()  const { return fCoordinates.M2();     }
       Scalar mag()   const { return fCoordinates.M();      }
       Scalar mt()    const { return fCoordinates.Mt();     }
       Scalar mt2()   const { return fCoordinates.Mt2();    }


       // Methods  requested by CMS ---
       Scalar energy() const { return fCoordinates.E();      }
       __device__ Scalar mass()   const { return fCoordinates.M();      }
       Scalar mass2()  const { return fCoordinates.M2();     }


       /**
          Methods setting a Single-component
          Work only if the component is one of which the vector is represented.
          For example SetE will work for a PxPyPzE Vector but not for a PxPyPzM Vector.
       */
       DeviceLorentzVector<CoordSystem>& SetE  ( Scalar a )  { fCoordinates.SetE  (a); return *this; }
       DeviceLorentzVector<CoordSystem>& SetEta( Scalar a )  { fCoordinates.SetEta(a); return *this; }
       DeviceLorentzVector<CoordSystem>& SetM  ( Scalar a )  { fCoordinates.SetM  (a); return *this; }
       DeviceLorentzVector<CoordSystem>& SetPhi( Scalar a )  { fCoordinates.SetPhi(a); return *this; }
       DeviceLorentzVector<CoordSystem>& SetPt ( Scalar a )  { fCoordinates.SetPt (a); return *this; }
       DeviceLorentzVector<CoordSystem>& SetPx ( Scalar a )  { fCoordinates.SetPx (a); return *this; }
       DeviceLorentzVector<CoordSystem>& SetPy ( Scalar a )  { fCoordinates.SetPy (a); return *this; }
       DeviceLorentzVector<CoordSystem>& SetPz ( Scalar a )  { fCoordinates.SetPz (a); return *this; }

    private:

       CoordSystem  fCoordinates;    // internal coordinate system
       static constexpr unsigned int fDimension = CoordinateType::Dimension;

    };  // DeviceLorentzVector<>



  // global methods

  /**
     Scale of a DeviceLorentzVector with a scalar quantity a
     \param a  scalar quantity of type a
     \param v  mathcore::DeviceLorentzVector based on any coordinate system
     \return a new mathcoreDeviceLorentzVector q = v * a same type as v
   */
    template< class CoordSystem >
    inline DeviceLorentzVector<CoordSystem> operator *
    ( const typename  DeviceLorentzVector<CoordSystem>::Scalar & a,
      const DeviceLorentzVector<CoordSystem>& v) {
       DeviceLorentzVector<CoordSystem> tmp(v);
       tmp *= a;
       return tmp;
    }

    // ------------- I/O to/from streams -------------

    //template< class char_t, class traits_t, class Coords >
    //inline
    //std::basic_ostream<char_t,traits_t> &
    //operator << ( std::basic_ostream<char_t,traits_t> & os
                  //, DeviceLorentzVector<Coords> const & v
       //)
    //{
       //if( !os )  return os;

       //typename Coords::Scalar a, b, c, d;
       //v.GetCoordinates(a, b, c, d);

       //if( detail::get_manip( os, detail::bitforbit ) )  {
        //detail::set_manip( os, detail::bitforbit, '\00' );
        //// TODO: call MF's bitwise-accurate functions on each of a, b, c, d
       //}
       //else  {
          //os << detail::get_manip( os, detail::open  ) << a
             //<< detail::get_manip( os, detail::sep   ) << b
             //<< detail::get_manip( os, detail::sep   ) << c
             //<< detail::get_manip( os, detail::sep   ) << d
             //<< detail::get_manip( os, detail::close );
       //}

       //return os;

    //}  // op<< <>()


     //template< class char_t, class traits_t, class Coords >
     //inline
     //std::basic_istream<char_t,traits_t> &
     //operator >> ( std::basic_istream<char_t,traits_t> & is
                   //, DeviceLorentzVector<Coords> & v
        //)
     //{
        //if( !is )  return is;

        //typename Coords::Scalar a, b, c, d;

        //if( detail::get_manip( is, detail::bitforbit ) )  {
           //detail::set_manip( is, detail::bitforbit, '\00' );
           //// TODO: call MF's bitwise-accurate functions on each of a, b, c
        //}
        //else  {
           //detail::require_delim( is, detail::open  );  is >> a;
           //detail::require_delim( is, detail::sep   );  is >> b;
           //detail::require_delim( is, detail::sep   );  is >> c;
           //detail::require_delim( is, detail::sep   );  is >> d;
           //detail::require_delim( is, detail::close );
        //}

        //if( is )
           //v.SetCoordinates(a, b, c, d);
        //return is;

     //}  // op>> <>()

#endif  // DEVICELORENTZVECTOR_H_
