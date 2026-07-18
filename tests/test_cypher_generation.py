import unittest
from unittest.mock import patch

from agents.cypher import (
    _parse_departure_constraint,
    generate_cypher_for_transport,
)
from agents.nodes import neo4j_query_node
from agents.state import SmartMoveState


class CypherGenerationTests(unittest.TestCase):
    def test_primary_combined_result_still_populates_schedule_and_fare_fallbacks(self):
        state: SmartMoveState = {
            "cypher_query": "MATCH (from:Place)-[s:Schedule]->(to:Place) RETURN s",
            "cypher_query_schedule": "MATCH (from:Place)-[s:Schedule]->(to:Place) RETURN s",
            "cypher_query_fare": "MATCH (from:Place)-[f:Fare]->(to:Place) RETURN f",
            "cypher_query_fare_reverse": "MATCH (from:Place)-[f:Fare]->(to:Place) RETURN f",
        }

        def fake_execute(query: str) -> str:
            if query == state["cypher_query"]:
                return '[{"origin": "Colombo", "destination": "Kandy"}]'
            if query == state["cypher_query_schedule"]:
                return '[{"schedule": "ok"}]'
            if query in {state["cypher_query_fare"], state["cypher_query_fare_reverse"]}:
                return '[{"fare": "ok"}]'
            return "[]"

        with patch("agents.nodes.execute_neo4j_query", side_effect=fake_execute):
            updated = neo4j_query_node(state)

        self.assertEqual(updated["result_source"], "combined")
        self.assertEqual(updated["result_schedule"], '[{"schedule": "ok"}]')
        self.assertEqual(updated["result_fare"], '[{"fare": "ok"}]')

    def test_between_morning_window_parses_to_range_constraint(self):
        constraint = _parse_departure_constraint("between 8 and 9 in the morning")
        self.assertEqual(
            constraint,
            {"op": "between", "start": "08:00", "end": "09:00"},
        )

    def test_between_morning_window_emits_schedule_where_bounds(self):
        state: SmartMoveState = {
            "user_query": (
                "I need to travel from Colombo to Kandy. "
                "Please provide all the details between 8 and 9 in the morning."
            ),
            "origin": "Colombo",
            "destination": "Kandy",
            "departure_time": "between 8 and 9 in the morning",
            "transport_type": None,
            "fare": None,
        }

        cypher = generate_cypher_for_transport(state)

        self.assertIn(
            'WHERE s.departure >= "08:00" AND s.departure <= "09:00"',
            cypher,
        )
        self.assertIn("ORDER BY s.departure ASC", cypher)
        self.assertNotIn(":Fare", cypher)

    def test_transport_queries_do_not_use_unsupported_schedule_properties_or_limit(self):
        state: SmartMoveState = {
            "user_query": "Bus to Kandy from Colombo tomorrow at 8am",
            "origin": "Colombo",
            "destination": "Kandy",
            "departure_time": "8am",
            "date": "tomorrow",
            "transport_type": "bus",
        }

        cypher = generate_cypher_for_transport(state)

        self.assertNotIn("s.transport_type", cypher)
        self.assertNotIn("s.service_type", cypher)
        self.assertNotIn("CONTAINS 'bus'", cypher)
        self.assertNotIn("LIMIT 5", cypher)
        self.assertIn('WHERE s.departure >= "08:00"', cypher)
        self.assertIn(
            "s { .arrival, .departure, .route_type, .service_type, .working_days, .fare_type } AS schedule_properties",
            cypher,
        )
        self.assertIn(
            "f { .fare, .route_type, .route_key, .service_type, .fare_type } AS fare_properties",
            cypher,
        )

    def test_greater_than_departure_time_orders_by_departure_asc(self):
        state: SmartMoveState = {
            "user_query": "Bus to Kandy from Colombo after 8am",
            "origin": "Colombo",
            "destination": "Kandy",
            "departure_time": "after 8am",
            "date": "tomorrow",
            "transport_type": "bus",
        }

        cypher = generate_cypher_for_transport(state)

        self.assertIn('WHERE s.departure >= "08:00"', cypher)
        self.assertIn("ORDER BY s.departure ASC", cypher)

    def test_combined_query_does_not_use_transport_properties(self):
        state: SmartMoveState = {
            "user_query": "Bus to Kandy from Colombo at 8am",
            "origin": "Colombo",
            "destination": "Kandy",
            "departure_time": "at 8am",
            "date": "tomorrow",
            "transport_type": "bus",
        }

        cypher = generate_cypher_for_transport(state)

        self.assertNotIn("transport_type", cypher)
        self.assertNotIn("service_type", cypher)
        self.assertNotIn("coalesce(s.transport_type", cypher)
        self.assertNotIn("CONTAINS 'bus'", cypher)


if __name__ == "__main__":
    unittest.main()
