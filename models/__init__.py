#from models.updown import UpDown
from models.xlan import XLAN
from models.xlan_hier import XLAN_HIER
from models.xlan_hier_suptopic_v6 import XLAN_HIER_SUPTOPIC_V6
from models.xlan_hier_suptopic_v6_abortall import XLAN_HIER_SUPTOPIC_V6_ABORTALL
from models.xlan_hier_suptopic_v6_SAP import XLAN_HIER_SUPTOPIC_V6_SAP
from models.xlan_hier_suptopic_v6_SAP_XE_topic import XLAN_HIER_SUPTOPIC_V6_SAP_XE_topic
from models.xlan_hier_suptopic_v6_SAP_rangeshrink_max import XLAN_HIER_SUPTOPIC_V6_SAP_rangeshrink_max
from models.updown import UpDown
from models.updown_baseline import UpDown_BASELINE
from models.updown_SAP import UpDown_SAP

__factory = {
    #'UpDown': UpDown,
    'XLAN': XLAN,
    'XLAN_HIER': XLAN_HIER,
    'XLAN_SUPTOPIC_V6': XLAN_HIER_SUPTOPIC_V6,
    'XLAN_SUPTOPIC_V6_ABORTALL': XLAN_HIER_SUPTOPIC_V6_ABORTALL,
    'XLAN_SUPTOPIC_V6_SAP': XLAN_HIER_SUPTOPIC_V6_SAP,
    'XLAN_SUPTOPIC_V6_SAP_XE_topic': XLAN_HIER_SUPTOPIC_V6_SAP_XE_topic,
    'XLAN_HIER_SUPTOPIC_V6_SAP_rangeshrink_max': XLAN_HIER_SUPTOPIC_V6_SAP_rangeshrink_max,
    'UPDOWN': UpDown,
    'UPDOWN_baseline': UpDown_BASELINE,
    'UpDown_SAP':UpDown_SAP,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)
