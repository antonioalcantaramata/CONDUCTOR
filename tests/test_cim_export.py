"""
The semantic layer as CIM RDF (backend/cim_export.py): valid RDF in both
formats, the CIM100 classes and references a CIM reader expects, and the
basis of each fact beside them — read back with rdflib and queried with SPARQL.
"""

import pandapower.networks as pn
import pandas as pd
import pytest

import cim_export as ce
import semantics as sm

rdflib = pytest.importorskip("rdflib")
P = "PREFIX cim: <http://iec.ch/TC57/CIM100#> PREFIX cdr: <urn:conductor:semantics#> "


@pytest.fixture(scope="module")
def graph():
    net = pn.create_cigre_network_mv(with_der="all")
    rows = pd.DataFrame({"substation_name": ["Bus 13"], "consumption": [0.02], "production": [0.0]})
    s = sm.build(net, provenance=sm.value_provenance(net, rows), network_id="cigre")
    xml, jl = ce.to_rdf_xml(s, "cigre"), ce.to_json_ld(s, "cigre")
    g = rdflib.Graph().parse(data=xml, format="xml")
    return g, rdflib.Graph().parse(data=jl, format="json-ld")


def _q(g, query):
    return [tuple(str(x) for x in row) for row in g.query(P + query)]


def test_both_formats_are_the_same_graph(graph):
    from rdflib.compare import isomorphic
    g, j = graph
    assert len(g) > 1000 and isomorphic(g, j)


def test_the_cim_classes_and_references(graph):
    g, _ = graph
    assert _q(g, "SELECT (COUNT(?u) AS ?n) WHERE { ?u a cim:PhotovoltaicUnit ; "
                 "cim:PowerElectronicsUnit.PowerElectronicsConnection ?c . ?c a cim:PowerElectronicsConnection }") \
        == [("8",)]
    assert _q(g, "SELECT ?n WHERE { ?s cim:Switch.normalOpen 'true' ; cim:IdentifiedObject.name ?n } ORDER BY ?n") \
        == [("S1",), ("S2",), ("S3",)]
    assert _q(g, "SELECT (COUNT(?n) AS ?c) WHERE { ?n a cim:TopologicalNode . "
                 "FILTER NOT EXISTS { ?n cim:TopologicalNode.BaseVoltage ?b } }") == [("0",)]
    # Every reference to an element resolves to a node of the graph.
    assert _q(g, "SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o . FILTER(isIRI(?o) && STRSTARTS(STR(?o), 'urn:uuid:')) "
                 "FILTER NOT EXISTS { ?o ?p2 ?o2 } }") == [("0",)]
    # A rotating unit has its generating unit; every unit's control has a mode.
    assert _q(g, "SELECT ?u WHERE { ?m a cim:SynchronousMachine ; cim:RotatingMachine.GeneratingUnit ?u . "
                 "?u a cim:ThermalGeneratingUnit }")
    assert _q(g, "SELECT (COUNT(?r) AS ?c) WHERE { ?r a cim:RegulatingControl . "
                 "FILTER NOT EXISTS { ?r cim:RegulatingControl.mode ?m } }") == [("0",)]


def test_the_basis_of_each_fact_travels_with_it(graph):
    g, _ = graph
    assert _q(g, "SELECT ?b WHERE { ?u cim:IdentifiedObject.name 'WKA 7' ; cdr:energySourceBasis ?b }") \
        == [("stated",)]
    assert _q(g, "SELECT ?k WHERE { ?l cim:IdentifiedObject.name 'Load CI13' ; cdr:qProvenance ?k }") \
        == [("derived",)]
    assert _q(g, "SELECT ?b WHERE { ?f a cim:Feeder ; cdr:basis ?b } LIMIT 1") == [("inferred",)]


def test_the_backend_serves_the_export(backend_client):
    r = backend_client.get("/api/network/cim", params={"format": "xml"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/rdf+xml")
    g = rdflib.Graph().parse(data=r.text, format="xml")
    assert _q(g, "SELECT (COUNT(?n) AS ?c) WHERE { ?n a cim:TopologicalNode }") == [("30",)]
    assert backend_client.get("/api/network/cim", params={"format": "jsonld"}).status_code == 200
    assert backend_client.get("/api/network/cim", params={"format": "pdf"}).status_code == 400


def test_mrids_are_stable_and_found(backend_client):
    full = backend_client.get("/api/network/semantics").json()
    bus = full["buses"][5]
    assert backend_client.get("/api/network/semantics").json()["buses"][5]["mrid"] == bus["mrid"]
    found = backend_client.get("/api/network/resolve", params={"name": bus["mrid"]}).json()
    assert found["element_type"] == "bus" and found["index"] == bus["index"] and found["how"] == "mrid"
    out = backend_client.get("/api/network/semantics", params={"about": "around", "element": bus["mrid"]}).json()
    assert out["element"]["kind"] == "bus" and out["element"]["name"] == bus["name"]
