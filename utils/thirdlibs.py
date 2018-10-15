from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ROOT
from ROOT import gInterpreter
from ROOT import gSystem

def load_delphes(delphes_dir):
    if hasattr(ROOT, "Delphes"):
        print("'delphes' was already loaded.")
        return None
    classes_dir = gSystem.ConcatFileName(delphes_dir, "classes")
    external_dir = gSystem.ConcatFileName(delphes_dir, "external")
    exrootanalysis_dir = gSystem.ConcatFileName(external_dir, "ExRootAnalysis")
    so_file = gSystem.ConcatFileName(delphes_dir, "libDelphes.so")

    gInterpreter.AddIncludePath(delphes_dir)
    gInterpreter.AddIncludePath(classes_dir)
    gInterpreter.AddIncludePath(external_dir)
    gInterpreter.AddIncludePath(exrootanalysis_dir)
    gSystem.Load(so_file)
    gInterpreter.Declare('#include "classes/DelphesClasses.h"')

