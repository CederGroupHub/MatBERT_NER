from os import path
import string
import regex
import unidecode
from monty.fractions import gcd_float
from chemdataextractor.doc import Paragraph
from gensim.models.phrases import Phraser
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition, CompositionError


class MaterialsTextTokenizer(object):
    def __init__(self, phraser_path):
        # initialize phraser from file
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.num_token = '<nUm>'
        self.phraser_path = phraser_path
        self.phraser = Phraser.load(phraser_path)
        # elements by symbol
        self.element = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
                        "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
                        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
                        "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
                        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
                        "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
                        "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Uue"]
        # elements by name
        self.element_name = ["hydrogen", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen", "fluorine",
                             "neon", "sodium", "magnesium", "aluminium", "silicon", "phosphorus", "sulfur", "chlorine", "argon",
                             "potassium", "calcium", "scandium", "titanium", "vanadium", "chromium", "manganese", "iron",
                             "cobalt", "nickel", "copper", "zinc", "gallium", "germanium", "arsenic", "selenium", "bromine",
                             "krypton", "rubidium", "strontium", "yttrium", "zirconium", "niobium", "molybdenum", "technetium",
                             "ruthenium", "rhodium", "palladium", "silver", "cadmium", "indium", "tin", "antimony", "tellurium",
                             "iodine", "xenon", "cesium", "barium", "lanthanum", "cerium", "praseodymium", "neodymium",
                             "promethium", "samarium", "europium", "gadolinium", "terbium", "dysprosium", "holmium", "erbium",
                             "thulium", "ytterbium", "lutetium", "hafnium", "tantalum", "tungsten", "rhenium", "osmium",
                             "iridium", "platinum", "gold", "mercury", "thallium", "lead", "bismuth", "polonium", "astatine",
                             "radon", "francium", "radium", "actinium", "thorium", "protactinium", "uranium", "neptunium",
                             "plutonium", "americium", "curium", "berkelium", "californium", "einsteinium", "fermium",
                             "mendelevium", "nobelium", "lawrencium", "rutherfordium", "dubnium", "seaborgium", "bohrium",
                             "hassium", "meitnerium", "darmstadtium", "roentgenium", "copernicium", "nihonium", "flerovium",
                             "moscovium", "livermorium", "tennessine", "oganesson", "ununennium"]
        # element names and capitalized element names
        self.element_name_ul = self.element_name+[en.capitalize() for en in self.element_name]
        # element symbols, element names, and capitalized element names
        self.element_and_name = self.element+self.element_name_ul
        # elements with valence state in parentheses
        self.element_valence_in_par = regex.compile(r"^("+r"|".join(self.element_and_name) +
                                                    r")(\(([IV|iv]|[Vv]?[Ii]{0,3})\))$")
        self.element_direction_in_par = regex.compile(r"^(" + r"|".join(self.element_and_name) + r")(\(\d\d\d\d?\))")
        # exactly IV, VI or has 2 consecutive II, or roman in parentheses
        self.valence_info = regex.compile(r"(II+|^IV$|^VI$|\(IV\)|\(V?I{0,3}\))")
        # units of measurement
        self.split_unit = ["K", "h", "V", "wt", "wt.", "MHz", "kHz", "GHz", "Hz", "days", "weeks",
                           "hours", "minutes", "seconds", "T", "MPa", "GPa", "at.", "mol.",
                           "at", "m", "N", "s-1", "vol.", "vol", "eV", "A", "atm", "bar",
                           "kOe", "Oe", "h.", "mWcm−2", "keV", "MeV", "meV", "day", "week", "hour",
                           "minute", "month", "months", "year", "cycles", "years", "fs", "ns",
                           "ps", "rpm", "g", "mg", "mAcm−2", "mA", "mK", "mT", "s-1", "dB",
                           "Ag-1", "mAg-1", "mAg−1", "mAg", "mAh", "mAhg−1", "m-2", "mJ", "kJ",
                           "m2g−1", "THz", "KHz", "kJmol−1", "Torr", "gL-1", "Vcm−1", "mVs−1",
                           "J", "GJ", "mTorr", "bar", "cm2", "mbar", "kbar", "mmol", "mol", "molL−1",
                           "MΩ", "Ω", "kΩ", "mΩ", "mgL−1", "moldm−3", "m2", "m3", "cm-1", "cm",
                           "Scm−1", "Acm−1", "eV−1cm−2", "cm-2", "sccm", "cm−2eV−1", "cm−3eV−1",
                           "kA", "s−1", "emu", "L", "cmHz1", "gmol−1", "kVcm−1", "MPam1",
                           "cm2V−1s−1", "Acm−2", "cm−2s−1", "MV", "ionscm−2", "Jcm−2", "ncm−2",
                           "Jcm−2", "Wcm−2", "GWcm−2", "Acm−2K−2", "gcm−3", "cm3g−1", "mgl−1",
                           "mgml−1", "mgcm−2", "mΩcm", "cm−2s−1", "cm−2", "ions", "moll−1",
                           "nmol", "psi", "mol·L−1", "Jkg−1K−1", "km", "Wm−2", "mass", "mmHg",
                           "mmmin−1", "GeV", "m−2", "m−2s−1", "Kmin−1", "gL−1", "ng", "hr", "w",
                           "mN", "kN", "Mrad", "rad", "arcsec", "Ag−1", "dpa", "cdm−2",
                           "cd", "mcd", "mHz", "m−3", "ppm", "phr", "mL", "ML", "mlmin−1", "MWm−2",
                           "Wm−1K−1", "Wm−1K−1", "kWh", "Wkg−1", "Jm−3", "m-3", "gl−1", "A−1",
                           "Ks−1", "mgdm−3", "mms−1", "ks", "appm", "ºC", "HV", "kDa", "Da", "kG",
                           "kGy", "MGy", "Gy", "mGy", "Gbps", "μB", "μL", "μF", "nF", "pF", "mF",
                           "A", "Å", "A˚", "μgL−1"]
        # number
        self.number_basic = regex.compile(r"^[+-]?\d*\.?\d+\(?\d*\)?+$", regex.DOTALL)
        # number with unit
        self.number_and_unit = regex.compile(r"^([+-]?\d*\.?\d+\(?\d*\)?+)([\p{script=Latin}|Ω|μ]+.*)", regex.DOTALL)
        # punctuation list
        self.punctuation = list(string.punctuation) + ["\"", "“", "”", "≥", "≤", "×"]
        # dictionary of element name : element symbol
        self.element_name_dict = {en: es for en, es in zip(self.element_name, self.element)}


    def tokenize(self, text, split_oxidation=True, keep_sentences=True):
        def split_token(token, split_oxidation=split_oxidation):
            ''' split token if it is a number with a common unit or an element with a valence state '''
            # check if element with valence state
            elem_with_valence = self.element_valence_in_par.match(token) if split_oxidation else None
            # check if number with unit
            number_unit = self.number_and_unit.match(token)
            if number_unit is not None and number_unit.group(2) in self.split_unit:
                # return split number and unit
                return [number_unit.group(1), number_unit.group(2)]
            elif elem_with_valence is not None:
                # return split element and valence state
                return [elem_with_valence.group(1), elem_with_valence.group(2)]
            else:
                # return unsplit token
                return [token]
        # tokenize
        chem_data_extractor_par = Paragraph(text)
        tokens = chem_data_extractor_par.tokens
        tokens_out = []
        for sentence in tokens:
            if keep_sentences:
                tokens_out.append([])
                for token in sentence:
                    tokens_out[-1] += split_token(token.text, split_oxidation=split_oxidation)
            else:
                for token in sentence:
                    tokens_out += split_token(token.text, split_oxidation=split_oxidation)
        return tokens_out


    def process(self, tokens,
                exclude_punctuation=False, convert_number=True, normalize_materials=True,
                remove_accents=True, make_phrases=False, split_oxidation=True, include_mat=False):
        # if string
        if not isinstance(tokens, list):
            return self.process(self.tokenize(tokens, split_oxidation=split_oxidation, keep_sentences=False),
                                exclude_punctuation=exclude_punctuation, convert_number=convert_number, normalize_materials=normalize_materials,
                                remove_accents=remove_accents, make_phrases=make_phrases, split_oxidation=split_oxidation)
        processed, mat_list = [], []
        for i, token in enumerate(tokens):
            # exclude punctuation
            if exclude_punctuation and token in self.punctuation:
                continue
            # convert number
            elif convert_number and self.is_number(token):
                try:
                    if (tokens[i-1] == "(" and tokens[i+1] == ')') or (tokens[i-1] == '〈' and tokens[i+1] == '〉'):
                        pass
                    else:
                        token = self.num_token
                except IndexError:
                    token = self.num_token
            # chemical element name
            elif token in self.element_name_ul:
                mat_list.append((token, self.element_name_dict[token.lower()]))
                token = token.lower()
            # simple formula
            elif self.is_simple_formula(token):
                normalized_formula = self.normalized_formula(token)
                mat_list.append((token, normalized_formula))
                if normalize_materials:
                    token = normalized_formula
            # lower case if only first letter is upper case
            elif (len(token) == 1 or (len(token) > 1 and token[0].isupper() and token[1:].islower())) and token not in self.element and self.element_direction_in_par.match(token) is None:
                token = token.lower()
            # remove accents
            if remove_accents:
                token = self.remove_accent(token)
            processed.append(token)
        # make phrases
        if make_phrases:
            processed = self.make_phrases(processed, reps=2)
        if include_mat:
            return processed, mat_list
        else:
            return processed
    

    def make_phrases(self, sentence, reps=2):
        # loop until repetitions are done
        while reps > 0:
            sentence = self.phraser[sentence]
            reps -= 1
        return sentence
    

    def is_number(self, x):
        return self.number_basic.match(x.replace(',', '')) is not None


    @staticmethod
    def is_element(txt):
        try:
            Element(txt)
            return True
        except ValueError:
            return False
    

    def is_simple_formula(self, text):
        if self.valence_info.search(text) is not None:
            return False
        elif any(char.isdigit() or char.islower() for char in text):
            try:
                if text in ['O2', 'N2', 'Cl2', 'F2', 'H2']:
                    return True
                composition = Composition(text)
                if len(composition.keys()) < 2 or any([not self.is_element(key) for key in composition.keys()]):
                    return False
                return True
            except(CompositionError, ValueError, OverflowError):
                return False
        else:
            return False
    

    @staticmethod
    def get_ordered_integer_formula(element_amount, max_denominator=1000):
        g = gcd_float(list(element_amount.values()), 1/max_denominator)
        d = {k: round(v/g) for k, v in element_amount.items()}
        formula = ''
        for k in sorted(d):
            if d[k] > 1:
                formula += k+str(d[k])
            elif d[k] != 0:
                formula += k
        return formula

    
    def normalized_formula(self, formula, max_denominator=1000):
        try:
            formula_dict = Composition(formula).get_el_amt_dict()
            return self.get_ordered_integer_formula(formula_dict, max_denominator)
        except(CompositionError, ValueError):
            return formula
    

    @staticmethod
    def remove_accent(text):
        return unidecode.unidecode(text) if len(text) > 1 else text