from typing import List, Tuple, Optional
import pandas as pd
from geopy.distance import geodesic

from pyspark.sql.types import Row

# function searches for points that lie within a (euclidean) ball of size *radius* around the query points
# if optional argument *k* is given then function searches for at most k nearest points that lie within
# a (euclidean) ball of size *radius* around the query points

# point = (latitude, longitude)
# df - Spark DataFrame collected in a list of Rows (in order to collect your DataFrame run: df.collect() )


def kRadiusNN(
    df: List[Row],
    radius: float,
    point: Tuple[float, float],
    label_column_name: str,
    k: Optional[int] = None,
) -> pd.DataFrame:
    assert (25.0625 <= point[0] <= 52.9375) and (
        -124.9375 <= point[1] <= -67.0625
    ), "Wrong coordinates (out of range)"

    neighbours_pd = pd.DataFrame(
        {"lon": [], "lat": [], "dist": [], label_column_name: []}
    )

    for row in df:

        lon = row["lon"]
        lat = row["lat"]
        label = row[label_column_name]
        dist = geodesic((lat, lon), point).km
        if dist <= radius:
            new_row = {"lon": lon, "lat": lat, "dist": dist, label_column_name: label}
            neighbours_pd.loc[len(neighbours_pd)] = new_row

    if k and (k < len(neighbours_pd)):

        neighbours_pd = neighbours_pd.sort_values("dist", ascending=True)
        if not (
            neighbours_pd.at[k - 1, "dist"] == neighbours_pd.at[k, "dist"]
        ):  # checking if there is no tie (more neighbours with the same distance)
            neighbours_pd = neighbours_pd.iloc[:k]
        else:
            raise Exception(
                "Unable to determine k nearest neighbours: more neighbours with the same distance"
            )

    if len(neighbours_pd) == 0:
        raise Exception("No neighbours found within the given radius")

    return neighbours_pd


# weighted: if True then function will weight points by the inverse of their distance (in this case, closer neighbours of
# a query point will have a greater influence than neighbors which are further away).


# point = (latitude, longitude)
# df - Spark DataFrame collected in a list of Rows (in order to collect your DataFrame run: df.collect(), before applying function to your DataFrame)
def predict_class(
    df: List[Row],
    point: Tuple[float, float],
    radius: float,
    label_column_name: str,
    k: Optional[int] = None,
    weighted: Optional[bool] = False,
) -> int:
    if k:
        neighbours = kRadiusNN(df, radius, point, label_column_name, k)
    else:
        neighbours = kRadiusNN(df, radius, point, label_column_name)

    if weighted:  # weighted nearest neighbours
        neighbours["dist"] = neighbours["dist"].map(lambda x: 1 / x)

        label0 = neighbours[neighbours[label_column_name] == 0]
        label1 = neighbours[neighbours[label_column_name] == 1]

        frequency0 = label0["dist"].sum()
        frequency1 = label1["dist"].sum()

        return 0 if frequency0 > frequency1 else 1

    else:

        if (
            len(neighbours[label_column_name].mode()) > 1
        ):  # this is only possible when the number of neighbours is an even number (since we perform binary classification)
            raise Exception(
                "Unable to predict the label: we have a tie, there is no clear winner in the majority voting"
            )
        else:
            predicted_label = neighbours[label_column_name].mode().iat[0]

        return predicted_label
