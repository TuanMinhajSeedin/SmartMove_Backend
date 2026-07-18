from agents.cypher import (
    _llm_cypher_respects_constraints,
    _parse_fare_intent,
    _parse_departure_constraint,
)

bad = """
MATCH (from:Place {name: "Colombo"})-[s:Schedule]->(to:Place {name: "Kandy"})
MATCH (from)-[f:Fare]->(to)
WITH from, to, s, f
ORDER BY f.fare ASC
RETURN x
"""
fi = _parse_fare_intent("LKR 500 to 800.")
op, t = _parse_departure_constraint("after 7 in the morning")
assert fi["mode"] == "range"
assert op == ">=" and t == "07:00"
assert (
    _llm_cypher_respects_constraints(
        bad, "both", op=op, time_24=t, fare_intent=fi, transport=""
    )
    is False
)
good = """
match (from:place {name: "colombo"})-[s:schedule]->(to:place {name: "kandy"})
where s.departure >= "07:00"
match (from)-[f:fare]->(to)
with from, to, s, f where f.fare >= 500.0 and f.fare <= 800.0
order by f.fare asc
return x
"""
assert (
    _llm_cypher_respects_constraints(
        good, "both", op=op, time_24=t, fare_intent=fi, transport=""
    )
    is True
)
print("ok")
