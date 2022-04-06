// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME dictDoubleCB
#define R__NO_DEPRECATION

/*******************************************************************/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define G__DICTIONARY
#include "RConfig.h"
#include "TClass.h"
#include "TDictAttributeMap.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TBuffer.h"
#include "TMemberInspector.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"
#include "TError.h"

#ifndef G__ROOT
#define G__ROOT
#endif

#include "RtypesImp.h"
#include "TIsAProxy.h"
#include "TFileMergeInfo.h"
#include <algorithm>
#include "TCollectionProxyInfo.h"
/*******************************************************************/

#include "TDataMember.h"

// Header files passed as explicit arguments
#include "RooDoubleCB.h"
#include "RooDoubleCB.h"

// Header files passed via #pragma extra_include

// The generated code does not explicitly qualify STL entities
namespace std {} using namespace std;

namespace ROOT {
   static void *new_RooDoubleCB(void *p = 0);
   static void *newArray_RooDoubleCB(Long_t size, void *p);
   static void delete_RooDoubleCB(void *p);
   static void deleteArray_RooDoubleCB(void *p);
   static void destruct_RooDoubleCB(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::RooDoubleCB*)
   {
      ::RooDoubleCB *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::RooDoubleCB >(0);
      static ::ROOT::TGenericClassInfo 
         instance("RooDoubleCB", ::RooDoubleCB::Class_Version(), "RooDoubleCB.h", 16,
                  typeid(::RooDoubleCB), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::RooDoubleCB::Dictionary, isa_proxy, 4,
                  sizeof(::RooDoubleCB) );
      instance.SetNew(&new_RooDoubleCB);
      instance.SetNewArray(&newArray_RooDoubleCB);
      instance.SetDelete(&delete_RooDoubleCB);
      instance.SetDeleteArray(&deleteArray_RooDoubleCB);
      instance.SetDestructor(&destruct_RooDoubleCB);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::RooDoubleCB*)
   {
      return GenerateInitInstanceLocal((::RooDoubleCB*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::RooDoubleCB*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

//______________________________________________________________________________
atomic_TClass_ptr RooDoubleCB::fgIsA(0);  // static to hold class pointer

//______________________________________________________________________________
const char *RooDoubleCB::Class_Name()
{
   return "RooDoubleCB";
}

//______________________________________________________________________________
const char *RooDoubleCB::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RooDoubleCB*)0x0)->GetImplFileName();
}

//______________________________________________________________________________
int RooDoubleCB::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RooDoubleCB*)0x0)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *RooDoubleCB::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RooDoubleCB*)0x0)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *RooDoubleCB::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RooDoubleCB*)0x0)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
void RooDoubleCB::Streamer(TBuffer &R__b)
{
   // Stream an object of class RooDoubleCB.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(RooDoubleCB::Class(),this);
   } else {
      R__b.WriteClassBuffer(RooDoubleCB::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_RooDoubleCB(void *p) {
      return  p ? new(p) ::RooDoubleCB : new ::RooDoubleCB;
   }
   static void *newArray_RooDoubleCB(Long_t nElements, void *p) {
      return p ? new(p) ::RooDoubleCB[nElements] : new ::RooDoubleCB[nElements];
   }
   // Wrapper around operator delete
   static void delete_RooDoubleCB(void *p) {
      delete ((::RooDoubleCB*)p);
   }
   static void deleteArray_RooDoubleCB(void *p) {
      delete [] ((::RooDoubleCB*)p);
   }
   static void destruct_RooDoubleCB(void *p) {
      typedef ::RooDoubleCB current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::RooDoubleCB

namespace {
  void TriggerDictionaryInitialization_dictDoubleCB_Impl() {
    static const char* headers[] = {
"RooDoubleCB.h",
0
    };
    static const char* includePaths[] = {
"/include",
"/home/dkondra/.conda/envs/cent7/5.3.1-py37/hmumu/include/",
"/home/dkondra/hmumu-coffea/lib/RooDoubleCB/",
0
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "dictDoubleCB dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_AutoLoading_Map;
class __attribute__((annotate(R"ATTRDUMP(Your description goes here...)ATTRDUMP"))) __attribute__((annotate("$clingAutoload$RooDoubleCB.h")))  RooDoubleCB;
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "dictDoubleCB dictionary payload"


#define _BACKWARD_BACKWARD_WARNING_H
// Inline headers
#include "RooDoubleCB.h"
#include "RooDoubleCB.h"
#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ nestedclasses;
#pragma link C++ class RooDoubleCB+;
#endif /* __CINT__ */

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[] = {
"RooDoubleCB", payloadCode, "@",
nullptr
};
    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("dictDoubleCB",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_dictDoubleCB_Impl, {}, classesHeaders, /*hasCxxModule*/false);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_dictDoubleCB_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_dictDoubleCB() {
  TriggerDictionaryInitialization_dictDoubleCB_Impl();
}
