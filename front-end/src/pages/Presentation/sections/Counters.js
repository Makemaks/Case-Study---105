/*
=========================================================
* Hate Speech and Offensive Language Dataset - v1.0
=========================================================
*/

import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";

// Material Kit 2 React components
import MKBox from "components/MKBox";

// Material Kit 2 React examples
import DefaultCounterCard from "examples/Cards/CounterCards/DefaultCounterCard";

function Counters() {
  return (
    <MKBox component="section" py={3}>
      <Container>
        <Grid container spacing={3} sx={{ mx: "auto" }}>
          <Grid item xs={12} md={3}>
            <DefaultCounterCard
              count={24783}
              title="Total Tweets"
              description="The dataset consists of 24,783 entries, capturing various types of speech."
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <DefaultCounterCard
              count={1430}
              title="Hate Speech"
              description="Around 1,430 tweets in the dataset are labeled as hate speech."
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <DefaultCounterCard
              count={19190}
              title="Offensive Language"
              description="Approximately 19,190 tweets are identified as containing offensive language."
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <DefaultCounterCard
              count={4163}
              title="Neither"
              description="There are around 4,163 tweets classified as neither hate speech nor offensive language."
            />
          </Grid>
        </Grid>
      </Container>
    </MKBox>
  );
}

export default Counters;
