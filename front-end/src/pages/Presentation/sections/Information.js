/*
=========================================================
* Hate Speech and Offensive Language Detection Service - v1.0
=========================================================
*/

import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";

// Material Kit 2 React components
import MKBox from "components/MKBox";

// Material Kit 2 React examples
import RotatingCard from "examples/Cards/RotatingCard";
import RotatingCardFront from "examples/Cards/RotatingCard/RotatingCardFront";
import RotatingCardBack from "examples/Cards/RotatingCard/RotatingCardBack";
import DefaultInfoCard from "examples/Cards/InfoCards/DefaultInfoCard";

// Images
import bgFront from "assets/images/rotating-card-bg-front.jpeg";
import bgBack from "assets/images/rotating-card-bg-back.jpeg";

function Information() {
  return (
    <MKBox component="section" py={6} my={6}>
      <Container>
        <Grid container item xs={11} spacing={3} alignItems="center" sx={{ mx: "auto" }}>
          <Grid item xs={12} lg={4} sx={{ mx: "auto" }}>
            <RotatingCard>
              <RotatingCardFront
                image={bgFront}
                icon="touch_app"
                title={
                  <>
                    Integrate with
                    <br />
                    Our Detection Model
                  </>
                }
                description="Easily integrate hate speech and offensive language detection into your website, app, or system using our simple API."
              />
              <RotatingCardBack
                image={bgBack}
                title="Discover More"
                description="Streamline your content moderation process by integrating our model. Quickly identify harmful content across multiple platforms."
                action={{
                  type: "internal",
                  route: "/sections/page-sections/page-headers",
                  label: "Start with API",
                }}
              />
            </RotatingCard>
          </Grid>
          <Grid item xs={12} lg={7} sx={{ ml: "auto" }}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <DefaultInfoCard
                  icon="content_copy"
                  title="Full Documentation"
                  description="Check out our comprehensive documentation to understand how to integrate the API and use our hate speech and offensive language detection service."
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <DefaultInfoCard
                  icon="flip_to_front"
                  title="Very Useful"
                  description="Our service is highly effective at detecting harmful language, ensuring that you can maintain a safe environment for your users."
                />
              </Grid>
            </Grid>
            <Grid container spacing={3} sx={{ mt: { xs: 0, md: 6 } }}>
              <Grid item xs={12} md={6}>
                <DefaultInfoCard
                  icon="price_change"
                  title="Save Time & Money"
                  description="Save development time and costs by utilizing our pre-built model and API. Get started quickly with minimal setup."
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <DefaultInfoCard
                  icon="devices"
                  title="Expand Your Imagination"
                  description="With our detection service, you can ensure a safe and respectful environment in your app, website, or system, unlocking endless possibilities for user interactions."
                />
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </Container>
    </MKBox>
  );
}

export default Information;
