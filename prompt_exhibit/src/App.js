import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import TablePage from "./custom/tablePage";

function App() {
  return (
    <Router>
      <Routes>
        <Route basename="table_page" path="/table" element={<TablePage />}>table_page</Route>
      </Routes>
    </Router>
  )
}

export default App;