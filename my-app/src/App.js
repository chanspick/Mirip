import { useEffect } from "react";
import {
  Routes,
  Route,
  useNavigationType,
  useLocation,
} from "react-router-dom";
import MiripHome from "./pages/MiripHome";
import MiripComp from "./pages/MiripComp";
import MiripEdu from "./pages/MiripEdu";

function App() {
  const action = useNavigationType();
  const location = useLocation();
  const pathname = location.pathname;

  useEffect(() => {
    if (action !== "POP") {
      window.scrollTo(0, 0);
    }
  }, [action, pathname]);

  useEffect(() => {
    let title = "";
    let metaDescription = "";

    switch (pathname) {
      case "/":
        title = "";
        metaDescription = "";
        break;
      case "/mirip-comp":
        title = "";
        metaDescription = "";
        break;
      case "/mirip-edu":
        title = "";
        metaDescription = "";
        break;
    }

    if (title) {
      document.title = title;
    }

    if (metaDescription) {
      const metaDescriptionTag = document.querySelector(
        'head > meta[name="description"]'
      );
      if (metaDescriptionTag) {
        metaDescriptionTag.content = metaDescription;
      }
    }
  }, [pathname]);

  return (
    <Routes>
      <Route path="/" element={<MiripHome />} />
      <Route path="/mirip-comp" element={<MiripComp />} />
      <Route path="/mirip-edu" element={<MiripEdu />} />
    </Routes>
  );
}
export default App;
