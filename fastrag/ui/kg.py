import networkx as nx
import pandas as pd
from pyvis.network import Network


def get_kg_html(relations):
    df = pd.DataFrame(relations, columns=["source", "target", "edge"]).drop_duplicates(
        subset=["source", "target"]
    )

    g_custom = nx.Graph()
    value_counts = pd.value_counts(list(df["source"]) + list(df["target"]))
    value_counts = value_counts[value_counts > 1]

    for node_name, node_count in value_counts.iteritems():
        g_custom.add_node(node_name, size=node_count + 5)  # size=15, , size=25 , shape='circle'

    value_counts_df = pd.DataFrame(value_counts).reset_index()[["index"]]
    df_min = df.join(
        value_counts_df.rename(columns={"index": "source"}).set_index("source"),
        on="source",
        how="inner",
    ).join(
        value_counts_df.rename(columns={"index": "target"}).set_index("target"),
        on="target",
        how="inner",
    )

    for _, df_row in df_min.iterrows():
        g_custom.add_edge(
            df_row["source"], df_row["target"], label=edge_type_to_label[df_row["edge"]]
        )

    nt = Network("1000px", "1000px", notebook=True)
    nt.from_nx(g_custom, show_edge_weights=True)
    nt.repulsion(node_distance=200, spring_length=500)
    nt.show_buttons()
    nt.show("nx.html")
    HtmlFile = open("nx.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    return source_code


edge_type_to_label = {
    "no_relation": "None",
    "org:alternate_names": "also called",
    "org:city_of_headquarters": "based in",
    "org:country_of_headquarters": "based in",
    "org:dissolved": "dissolved",
    "org:founded": "founded",
    "org:founded_by": "founded by",
    "org:member_of": "is in",
    "org:members": "members in",
    "org:number_of_employees/members": "number",
    "org:parents": "parent of",
    "org:political/religious_affiliation": "affiliated with",
    "org:shareholders": "shareholder",
    "org:stateorprovince_of_headquarters": "based in",
    "org:subsidiaries": "part of",
    "org:top_members/employees": "top of",
    "org:website": "website",
    "per:age": "aged",
    "per:alternate_names": "also called",
    "per:cause_of_death": "died of",
    "per:charges": "charged with",
    "per:children": "child of",
    "per:cities_of_residence": "lives in",
    "per:city_of_birth": "born in",
    "per:city_of_death": "died in",
    "per:countries_of_residence": "lives in",
    "per:country_of_birth": "born in",
    "per:country_of_death": "died in",
    "per:date_of_birth": "date of birth",
    "per:date_of_death": "date of death",
    "per:employee_of": "works at",
    "per:origin": "origin",
    "per:other_family": "family",
    "per:parents": "parent of",
    "per:religion": "believes in",
    "per:schools_attended": "attended",
    "per:siblings": "sibling of",
    "per:spouse": "married to",
    "per:stateorprovince_of_birth": "born in",
    "per:stateorprovince_of_death": "died in",
    "per:stateorprovinces_of_residence": "lives in",
    "per:title": "title",
}
