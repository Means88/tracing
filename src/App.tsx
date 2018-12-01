import * as React from 'react';
import './App.css';
import Monaco from "./components/monaco";
import TraceBody from "./components/trace-body";
import TraceHeaderSelection from "./components/trace-header-selection";
import TraceStore from "./stores/TraceStore";

class App extends React.Component {
  public componentDidMount() {
    TraceStore.loadTraces();
  }

  public render() {
    return (
      <div className="App">
        <Monaco />
        <TraceHeaderSelection />
        <TraceBody />
      </div>
    );
  }
}

export default App;
