import PropTypes from "prop-types";
import styles from "./Credentials.module.css";

const Credentials = ({ label, placeholder, type = "text" }) => {
  return (
    <div className={styles.credentials}>
      <label className={styles.label}>{label}</label>
      <input
        className={styles.input}
        type={type}
        placeholder={placeholder}
      />
    </div>
  );
};

Credentials.propTypes = {
  label: PropTypes.string.isRequired,
  placeholder: PropTypes.string.isRequired,
  type: PropTypes.string,
};

export default Credentials;
