import Credentials from "./Credentials";
import Button from "./Button";
import FormLayout from "./FormLayout";
import styles from "./LoginForm.module.css";

const LoginForm = () => {
  return (
    <FormLayout
      title="Welcome !"
      subtitle="Sign in to Lorem Ipsum is simply"
    >
      <Credentials label="User name" placeholder="Enter your user name" />
      <Credentials label="Password" placeholder="Enter your password" type="password" />
      
      <div className={styles.rememberForgot}>
        <div className={styles.rememberMe}>
          <input type="checkbox" />
          <label>Remember me</label>
        </div>
        <div className={styles.forgotPassword}>Forgot Password ?</div>
      </div>

      <Button text="Login" onClick={() => console.log("Login clicked")} />

      <div className={styles.registerLink}>
        Don’t have an Account? <a href="/register">Register</a>
      </div>
    </FormLayout>
  );
};

export default LoginForm;
